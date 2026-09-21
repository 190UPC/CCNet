import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import to_2tuple, trunc_normal_

from collections import OrderedDict

import numbers
from pdb import set_trace as stx

from einops import rearrange

# for idynamic
from .idynamicdwconv_util import *

import torch.nn.functional as F
class InceptionDWConv(nn.Module):
    def __init__(self, dim, bias, kernel_size=3):
        super().__init__()
        self.chunk = dim // 2  # 对半分成2份
        p = kernel_size // 2
        g = self.chunk

        self.branch1 = nn.Conv2d(self.chunk, self.chunk, kernel_size,
                                 padding=p, groups=g, bias=bias)
        self.branch2 = nn.Sequential(
            nn.Conv2d(self.chunk, self.chunk, kernel_size,
                      padding=p, groups=g, bias=bias),
            nn.Conv2d(self.chunk, self.chunk, kernel_size,
                      padding=p, groups=g, bias=bias),
        )

    def forward(self, x):
        c = self.chunk
        return torch.cat([
            self.branch1(x[:,  :c, :, :]),
            self.branch2(x[:, c:, :, :]),
        ], dim=1)

class QKSubbandBias(nn.Module):
    """从 x_dwt 四个子带生成 Q/K 的 scale 和 shift"""
    def __init__(self, dim, num_heads):
        super().__init__()
        # x_dwt 通道数 = dim (四个子带拼接后)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_heads * 2),  # 每个head一组 scale/shift
        )
        self.num_heads = num_heads

    def forward(self, x_dwt):
        b, c, h, w = x_dwt.shape
        feat = self.pool(x_dwt).flatten(1)       # (b, c)
        out = self.mlp(feat)                      # (b, num_heads*2)
        scale, shift = out.chunk(2, dim=1)        # 各 (b, num_heads)
        # reshape 为可广播的形状 (b, heads, 1, 1)
        scale = (1 + scale.view(b, self.num_heads, 1, 1))  # 残差初始化，接近1
        shift = shift.view(b, self.num_heads, 1, 1)
        return scale, shift

class SubbandCrossModulation(nn.Module):
    """
    让4个子带(LL/LH/HL/HH)互相生成调制权重
    LL <-> HH : 低频结构指导高频细节
    LH <-> HL : 水平/垂直边缘互相增强
    """
    def __init__(self, dim):
        super().__init__()
        c = dim // 4  # 每个子带的通道数

        # 每个子带生成调制系数的轻量分支
        # GlobalAvgPool + GlobalMaxPool -> MLP -> 空间权重
        self.ll_to_hh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c, bias=False),
            nn.GELU(),
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )
        self.hh_to_ll = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c, bias=False),
            nn.GELU(),
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )
        self.lh_to_hl = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c, bias=False),
            nn.GELU(),
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )
        self.hl_to_lh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c, bias=False),
            nn.GELU(),
            nn.Linear(c, c, bias=False),
            nn.Sigmoid()
        )

        # 融合后的轻量卷积细化
        self.refine = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, attn_out, x_dwt):
        c = x_dwt.shape[1] // 4
        ll, lh, hl, hh = x_dwt.chunk(4, dim=1)  # 各子带 (b, c, h, w)

        # 生成交叉调制权重 (b, c) -> (b, c, 1, 1)
        w_ll = self.hh_to_ll(hh).view(-1, c, 1, 1)  # HH指导LL
        w_hh = self.ll_to_hh(ll).view(-1, c, 1, 1)  # LL指导HH
        w_lh = self.hl_to_lh(hl).view(-1, c, 1, 1)  # HL指导LH
        w_hl = self.lh_to_hl(lh).view(-1, c, 1, 1)  # LH指导HL

        # 对 attention_out 的对应子带施加调制
        out_ll, out_lh, out_hl, out_hh = attn_out.chunk(4, dim=1)
        out_ll = out_ll * w_ll
        out_lh = out_lh * w_lh
        out_hl = out_hl * w_hl
        out_hh = out_hh * w_hh

        out = torch.cat([out_ll, out_lh, out_hl, out_hh], dim=1)
        out = self.refine(out)  # 深度卷积细化
        return out
# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):    # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# FFN
# class FeedForward(nn.Module):
#     """
#         GDFN in Restormer: [github] https://github.com/swz30/Restormer
#     """
#     def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
#         super(FeedForward, self).__init__()

#         self.input_resolution = input_resolution
#         self.dim = dim
#         self.ffn_expansion_factor = ffn_expansion_factor

#         hidden_features = int(dim*ffn_expansion_factor)
#         self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x)
#         return x
class FeedForward(nn.Module):
    """
    GDFN + InceptionDW: replaces the single 3x3 dwconv with a
    4-branch multi-scale depthwise conv; everything else unchanged.
    """
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super().__init__()
        self.input_resolution  = input_resolution
        self.dim               = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in  = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # ← only this line changes vs. original
        self.dwconv      = InceptionDWConv(hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        # base feed forward network in SwinIR
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        return self.body(x)
# ---------------------------------------------------------------------------------------------------------------------

class SMA(nn.Module):
    """
        SparseGSA is based on MDTA
        MDTA in Restormer: [github] https://github.com/swz30/Restormer
        TLC: [github] https://github.com/megvii-research/TLC
        We use TLC-Restormer in forward function and only use it in test mode
    """
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SMA, self).__init__()
        self.tlc_flag = tlc_flag
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Q/K 频域增强门控
        self.freq_gate = nn.Parameter(torch.ones(1, num_heads, dim // num_heads, 1))

        # ---- 新增：频域残差支路 ----
        # 可学习的复数频域滤波器（实部 + 虚部分开存）
        # shape: (1, C, 1, 1) 广播到频谱的所有空间位置
        self.freq_filter_real = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.freq_filter_imag = nn.Parameter(torch.zeros(1, dim, 1, 1))
        # 融合权重：主路 vs 频域支路，可学习标量
        self.freq_branch_weight = nn.Parameter(torch.zeros(1))  # 初始化为0，训练后自适应
        # ---- 新增结束 ----

        self.act = nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        self.kernel_size = [tlc_kernel, tlc_kernel]

    def freq_branch(self, x):
        """
        纯频域残差支路：
        X -> FFT -> 复数滤波 -> IFFT -> 残差输出
        """
        b, c, h, w = x.shape
        # 转频域
        f = torch.fft.rfft2(x, norm='ortho')                        # (b, c, h, w//2+1) 复数
        # 复数滤波：(a+bj)(c+dj) = (ac-bd) + (ad+bc)j
        f_real = f.real * self.freq_filter_real - f.imag * self.freq_filter_imag
        f_imag = f.real * self.freq_filter_imag + f.imag * self.freq_filter_real
        f_filtered = torch.complex(f_real, f_imag)
        # 转回空域
        out = torch.fft.irfft2(f_filtered, s=(h, w), norm='ortho')  # (b, c, h, w)
        return out  # 残差，由外部加回 x

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        # Q/K 频域增强
        def freq_enhance(feat):
            b, c, h, w = feat.shape
            f = torch.fft.rfft2(feat, norm='ortho')
            gate = torch.sigmoid(self.freq_gate).reshape(1, c, 1, 1)
            f = f * gate
            feat_enhanced = torch.fft.irfft2(f, s=(h, w), norm='ortho')
            return feat + feat_enhanced

        q = freq_enhance(q)
        k = freq_enhance(k)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            out = self.project_out(out)
        else:
            qkv = self.grids(qkv)
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
            out = self.grids_inverse(out)
            out = self.project_out(out)

        # ---- 新增：融合频域残差支路 ----
        freq_out = self.freq_branch(x)                               # 频域支路，基于原始输入 x
        weight = torch.sigmoid(self.freq_branch_weight)              # 映射到 (0,1)
        out = out + weight * freq_out                                # 加权融合
        # ---- 新增结束 ----

        return out

    # grids / grids_inverse 不变，省略...
    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w) # [1, 48, 72, 72]
        # print("self.original_size", self.original_size)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size # [1, 48, 72, 72]
        # print("self.original_size", self.original_size)

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------

class AttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(AttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SMA(dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x
# ---------------------------------------------------------------------------------------------------------------------

import pywt
import numpy as np
from torch.autograd import Function

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class WMA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(WMA, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test
        self.qk_bias = QKSubbandBias(dim, num_heads)
        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwt = DWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        self.filter = SubbandCrossModulation(dim)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=bias)
        self.idwt_sa = IDWT_2D(wave='haar')

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, x):
        x_dwt = self.dwt(self.reduce(x))

        qkv = self.qkv_dwconv(self.qkv(x_dwt))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # ── 新增：子带仿射偏置 ──────────────────────────
        scale, shift = self.qk_bias(x_dwt)   # 由四子带全局统计生成
        q = q * scale + shift
        k = k * scale + shift
        # ───────────────────────────────────────────────

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)

        return out, x_dwt

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = x

        if self.training or not self.tlc_flag:
            out, x_dwt = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)
            out = self.filter(out, x_dwt)
            out = self.idwt_sa(out)
            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out, x_dwt = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2]//2, w=qkv.shape[-1]//2)
        out = self.filter(out, x_dwt)
        out = self.idwt_sa(out)
        out = self.project_out(out)

        out = self.grids_inverse(out)  # reverse

        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            # print(preds.size())
            # print(outs.size())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


class WaveAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(WaveAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = WMA(dim=dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------

# BuildBlocks
class BuildBlock(nn.Module):
    # Sorry for the redundant parameter setting
    # it is easier for ablation study while during experiment
    # if necessary it can be changed to **args
    def __init__(self, dim, blocks=3, buildblock_type='edge',
                 window_size=7,restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None
                 ):
        super(BuildBlock, self).__init__()

        self.input_resolution = input_resolution

        # those all for extra_repr
        # --------
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.window_size = window_size
        self.tlc = tlc_flag
        # ---------

        # buildblock body
        # ---------
        body = []
        if buildblock_type == 'Wave':  #this
            for _ in range(blocks):
                body.append(AttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))
                body.append(WaveAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))
        # --------HybridAttentionBlock

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))   # as like SwinIR, we use one Conv3x3 layer after buildblock
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x     # shortcut in buildblock

    def extra_repr(self) -> str:
        return f'dim={self.dim}, blocks={self.blocks}, buildblock_type={self.buildblock_type}, ' \
               f'window_size={self.window_size}, tlc={self.tlc}'
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

       but for our model, we give up Traditional Upsample and use UpsampleOneStep for better performance not only in
       lightweight SR model, Small/XSmall SR model, but also for our base model.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


# Traditional Upsample from SwinIR EDSR RCAN
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Network
@ARCH_REGISTRY.register()
class ARCH555Net(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 90
        depths (tuple(int)): Depth of each BuildBlock
        num_heads (tuple(int)): Number of attention heads in different layers
        window_size (int): Window size. Default: 7
        ffn_expansion_factor (float): Ratio of feedforward network hidden dim to embedding dim. Default: 2
        ffn_type (str): feedforward network type, such as GDFN and BaseFFN
        bias (bool): If True, add a learnable bias to layers. Default: True
        body_norm (bool): Normalization layer. Default: False
        idynamic (bool): using idynamic for local attention. Default: True
        tlc_flag (bool): using TLC during validation and test. Default: True
        tlc_kernel (int): TLC kernel_size [x2, x3, x4] -> [96, 72, 48]
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'
    """

    def __init__(self,
                 in_chans=3,
                 dim=48,
                 groups=3,
                 blocks=3,
                 buildblock_type='Wave',
                 window_size=7,
                 restormer_num_heads=8, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=64, activation='relu',
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 body_norm=False,
                 input_resolution=None,     # input_resolution = (height, width)
                 **kwargs):
        super(ARCH555Net, self).__init__()

        # for flops counting
        self.dim = dim
        self.input_resolution = input_resolution

        # MeanShift for Image Input
        # ---------
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # -----------

        # Upsample setting
        # -----------
        self.upscale = upscale
        self.upsampler = upsampler
        # -----------

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # the overlap_embed: remember to set it into bias=False
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # ------------------------- 2, deep feature extraction ------------------------- #
        m_body = []

        # Base on the Transformer, When we use pre-norm we need to build a norm after the body block
        if body_norm:       # Base on the SwinIR model, there are LayerNorm Layers in PatchEmbed Layer between body
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks, buildblock_type,
                 window_size, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        m_body.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)))

        self.deep_feature_extraction = nn.Sequential(*m_body)

        # ------------------------- 3, high quality image reconstruction ------------------------- #

        # setting for pixelshuffle for big model, but we only use pixelshuffledirect for all our model
        # -------
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans
        # -------

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        pass    # all are in forward function including deep feature extraction

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        else:
            # for image denoising and JPEG compression artifact reduction
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x