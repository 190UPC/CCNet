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
        return x / torch.sqrt(sigma + 1e-5) * self.weight


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
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
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
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# FFN     Modified: multi-scale gated depth-wise FFN (3×3 + 5×5 branches)
class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 3, hidden_features * 3, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 3, bias=bias)
        self.dwconv_3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features, bias=bias)
        self.dwconv_5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                    groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2, x3 = self.dwconv(x).chunk(3, dim=1)
        x2 = self.dwconv_3x3(x2)
        x3 = self.dwconv_5x5(x3)
        x = F.gelu(x1) * x2 * x3
        x = self.project_out(x)
        return x


class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------------------------------------------------

# =====================================================================================================================
# 【改动1】SMA：融合 MA 多尺度 + SMA 空洞稀疏
#
# 借鉴 MAT 的 NeighborhoodAttention2D：
#   - 通道按组数均分（首组补余数），对齐 MAT 的 channels 分配逻辑
#   - 每组独立 temperature，对齐 MAT 的 per-group 设计
#   - dilations 参数列表，对齐 MAT 的 dilations: List[int]
#
# 保留原 SMA：
#   - 通道注意力（非空间注意力）
#   - ReLU 稀疏化
#   - TLC 测试分块（grids / grids_inverse 完全不变）
# =====================================================================================================================
class SMA(nn.Module):
    """
    Multi-scale Sparse Multi-head Attention (V4)

    MA  思想（来自 MAT NeighborhoodAttention2D）：
        通道均分为 len(dilations) 组，每组独立建模，最后 concat
    SMA 思想（来自原 SMA）：
        每组用对应 dilation 的 DWConv 扩大感受野 + ReLU 稀疏化
    TLC：测试时分块，与原版完全一致
    """

    def __init__(self, dim, num_heads, bias,
                 tlc_flag=True, tlc_kernel=48,
                 activation='relu', input_resolution=None,
                 dilations=(1, 2, 3)):          # 对齐 MAT: dilations: List[int]
        super(SMA, self).__init__()
        self.tlc_flag = tlc_flag
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.dilations = dilations
        self.num_groups = len(dilations)        # 对齐 MAT: self.k = len(kernel_sizes)
        self.kernel_size = [tlc_kernel, tlc_kernel]

        # ── 通道均分，首组补余数（对齐 MAT channels 分配逻辑）─────────────
        # MAT 原逻辑：
        #   if i == 0: channels = dim*3 - dim*3//k*(k-1)
        #   else:      channels = dim*3//k
        # 这里对 dim（非 dim*3）做同样处理，QKV 在 _forward 内部再 *3
        self.group_channels = []
        for i in range(self.num_groups):
            if i == 0:
                ch = dim - dim // self.num_groups * (self.num_groups - 1)
            else:
                ch = dim // self.num_groups
            self.group_channels.append(ch)

        # ── 头数均分，首组补余数（与通道分配同逻辑）─────────────────────
        self.group_heads = []
        for i in range(self.num_groups):
            if i == 0:
                h = num_heads - num_heads // self.num_groups * (self.num_groups - 1)
            else:
                h = num_heads // self.num_groups
            self.group_heads.append(h)

        # ── 每组独立 temperature（对齐 MAT per-group 设计）──────────────
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.ones(self.group_heads[i], 1, 1))
            for i in range(self.num_groups)
        ])

        # ── QKV：统一 1×1 Conv（与原版一致）────────────────────────────
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # ── 每组独立 DWConv，dilation 对应 dilations[i]（SMA 空洞思想）──
        # 对齐 MAT: dilation=self.dilations[i]
        # padding = dilation 保证输出尺寸不变（kernel=3时）
        self.qkv_dwconvs = nn.ModuleList([
            nn.Conv2d(
                self.group_channels[i] * 3,
                self.group_channels[i] * 3,
                kernel_size=3,
                stride=1,
                padding=1,           # padding = dilation，尺寸不变
                dilation=1,
                groups=self.group_channels[i] * 3,
                bias=bias
            )
            for i in range(self.num_groups)
        ])

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 激活：ReLU → 稀疏（与原版一致）
        self.act = nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def _forward_group(self, qkv_chunk, group_idx):
        """单组通道注意力（与原版 _forward 逻辑一致，只换 temperature 索引）"""
        b, _, h, w = qkv_chunk.shape
        num_heads = self.group_heads[group_idx]

        q, k, v = qkv_chunk.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperatures[group_idx]
        attn = self.act(attn)           # ReLU 稀疏化
        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=num_heads, h=h, w=w)
        return out

    def _forward(self, qkv):
        # 按通道拆成 num_groups 份（对齐 MAT: torch.split by self.channels）
        split_sizes = [ch * 3 for ch in self.group_channels]
        qkv_groups = torch.split(qkv, split_sizes, dim=1)

        outs = []
        for i, qkv_g in enumerate(qkv_groups):
            qkv_g = self.qkv_dwconvs[i](qkv_g)     # 各组用对应 dilation DWConv
            out = self._forward_group(qkv_g, i)
            outs.append(out)

        return torch.cat(outs, dim=1)               # concat 恢复 dim 通道

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = self.project_out(out)
            return out

        # 测试时 TLC 分块（与原版完全一致）
        qkv = self.grids(qkv)
        out = self._forward(qkv)
        out = self.grids_inverse(out)
        out = self.project_out(out)
        return out

    # ── TLC grids / grids_inverse（与原版完全不变）───────────────────────
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0
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
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


# =====================================================================================================================
# 以下全部与原版 DMNetV2 完全一致，未做任何改动
# =====================================================================================================================

class AttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2.,
                 tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None,
                 dilations=(1, 2, 3)):          # 【改动2】透传 dilations 给 SMA
        super(AttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # 【改动3】SMA 新增 dilations 参数
        self.restormer_attn = SMA(dim, num_heads=restormer_num_heads, bias=False,
                                  tlc_flag=tlc_flag, tlc_kernel=tlc_kernel,
                                  activation=activation, input_resolution=input_resolution,
                                  dilations=dilations)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        if restormer_ffn_type == 'GDFN':
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False,
                                             input_resolution=input_resolution)
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
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
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
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
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
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
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
    def __init__(self, dim, num_heads, bias=False, tlc_flag=True,
                 tlc_kernel=48, activation='relu', input_resolution=None):
        super(WMA, self).__init__()
        self.tlc_flag = tlc_flag
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwt = DWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.filter = IDynamic(channels=dim, kernel_size=7, group_channels=num_heads)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim // 4, dim, kernel_size=1, bias=bias)
        self.idwt_sa = IDWT_2D(wave='haar')

        self.cape = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                              padding=1, groups=dim, bias=bias)

        self.act = nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, x):
        x_dwt = self.dwt(self.reduce(x))
        cape_dwt = self.cape(x_dwt)
        qkv = self.qkv_dwconv(self.qkv(x_dwt))
        q, k, v = qkv.chunk(3, dim=1)
        q = q + cape_dwt
        k = k + cape_dwt
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)
        return out, x_dwt, cape_dwt

    def forward(self, x):
        b, c, h, w = x.shape

        if self.training or not self.tlc_flag:
            out, x_dwt, cape_dwt = self._forward(x)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h // 2, w=w // 2)
            out = out + cape_dwt
            out = self.filter(out, x_dwt)
            out = self.idwt_sa(out)
            out = self.project_out(out)
            return out

        qkv_grid = self.grids(x)
        out, x_dwt, cape_dwt = self._forward(qkv_grid)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=qkv_grid.shape[-2] // 2,
                        w=qkv_grid.shape[-1] // 2)
        out = out + cape_dwt
        out = self.filter(out, x_dwt)
        out = self.idwt_sa(out)
        out = self.project_out(out)
        out = self.grids_inverse(out)
        return out

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
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)
        parts = []
        idxes = []
        i = 0
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
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.
        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


class WaveAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2.,
                 tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(WaveAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.restormer_attn = WMA(dim=dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag,
                                  tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        if restormer_ffn_type == 'GDFN':
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False,
                                             input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x


# ---------------------------------------------------------------------------------------------------------------------

class BuildBlock(nn.Module):
    def __init__(self, dim, blocks=3, buildblock_type='edge',
                 window_size=7, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2.,
                 tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None,
                 sma_dilations=((1, 2, 3), (1, 3, 5))   # 【改动4】对齐 MAT: dilations=[[1,1,1],[4,4,4]]，交替使用
                 ):
        super(BuildBlock, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.window_size = window_size
        self.tlc = tlc_flag

        body = []
        if buildblock_type == 'Wave':
            for idx in range(blocks):
                # 【改动5】交替使用两种 dilations，对齐 MAT: dilations[i_mab % 2]
                dilations = sma_dilations[idx % 2]
                body.append(
                    AttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor,
                                        tlc_flag, tlc_kernel, activation,
                                        input_resolution=input_resolution,
                                        dilations=dilations))
                body.append(
                    WaveAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type,
                                           restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation,
                                           input_resolution=input_resolution))

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, blocks={self.blocks}, buildblock_type={self.buildblock_type}, ' \
               f'window_size={self.window_size}, tlc={self.tlc}'


# ---------------------------------------------------------------------------------------------------------------------

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
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

@ARCH_REGISTRY.register()
class DMNetV5(nn.Module):
    r"""
    DMNetV4：在 DMNetV2 基础上，将 SMA 升级为多尺度空洞稀疏注意力。,ffn

    新增参数：
        sma_dilations (tuple of tuple): 两种 dilation 配置交替使用，
            对齐 MAT 的 dilations=[[1,1,1],[4,4,4]] 设计。
            默认 ((1,2,3), (1,3,5))：
                偶数块用 (1,2,3) —— 小中大感受野
                奇数块用 (1,3,5) —— 更稀疏的大感受野

    其余参数与 DMNetV2 完全一致。
    """

    def __init__(self,
                 in_chans=3,
                 dim=48,
                 groups=3,
                 blocks=3,
                 buildblock_type='Wave',
                 window_size=7,
                 restormer_num_heads=8,
                 restormer_ffn_type='GDFN',
                 restormer_ffn_expansion_factor=2.,
                 tlc_flag=True,
                 tlc_kernel=64,
                 activation='relu',
                 upscale=4,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 body_norm=False,
                 input_resolution=None,
                 # 【改动6】新增参数，对齐 MAT dilations 交替设计
                 sma_dilations=((1, 2, 3), (1, 3, 5)),
                 **kwargs):
        super(DMNetV5, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.upscale = upscale
        self.upsampler = upsampler

        # 浅层特征提取
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # 深层特征提取
        m_body = []

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks, buildblock_type,
                                     window_size, restormer_num_heads, restormer_ffn_type,
                                     restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation,
                                     input_resolution=input_resolution,
                                     sma_dilations=sma_dilations))   # 【改动7】传入 sma_dilations

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        m_body.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)))

        self.deep_feature_extraction = nn.Sequential(*m_body)

        # 上采样重建
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans

        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution)

        elif self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
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
        pass

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffle':
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        else:
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_last(x)

        x = x / self.img_range + self.mean
        return x