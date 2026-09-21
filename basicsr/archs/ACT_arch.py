"""
Asymmetric Content-aided Transformer for Efficient Image Super-Resolution
Qian Wang, Yanyu Mao, Ruilong Guo, Yao Tang, Jing Wei, Bo Quan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


###### split image to windows ######
def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

###### reverse windows to image ######
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return img


def Pixelshuffle_Block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (kernel_size, kernel_size), (stride, stride),
                     padding='same')
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


######  Efficient Feed-Forward Network  ######
class EFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwc1 = InceptionDWG(hidden_features, hidden_features, 3, 'same')
        # self.dwc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.dwc1(self.fc1(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.act(x)
        x = self.drop(x)
        return x


###### Efficient Feature Enhanced Module ######
class EFEM(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.dim = in_dim // 4
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.dwc = InceptionDWG(in_dim, in_dim, 3, 'same')
        self.conv1x1_2 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.dwc(out)
        out = self.conv1x1_2(self.act(out))
        return out


######  Inception Depth-Wise Convolution Group  ######
class InceptionDWG(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.dim = in_dim // 4
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)

    def forward(self, input):
        out = input
        out = torch.cat([self.conv2_1(out[:, :self.dim, :, :]), self.conv2_2(out[:, self.dim:self.dim * 2, :, :]),
                         self.conv2_3(out[:, self.dim * 2:self.dim * 3, :, :]),
                         self.conv2_4(out[:, self.dim * 3:, :, :])], dim=1)
        return out


######  Residual EFEM  ######
class R_EFEM(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding='same'):
        super().__init__()
        self.conv1 = EFEM(in_dim, out_dim, kernel_size, padding)
        # self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input) + input
        return out


######  Content-aided Self-Attention  ######
class CaSA(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=4, dim_out=None, num_heads=1, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx==1:
            W_sp, H_sp = self.resolution, self.split_size
        elif idx==2:
            W_sp, H_sp = self.resolution//2, self.resolution//2
        else:
            W_sp, H_sp = self.resolution, self.resolution
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_qk = nn.Conv2d(dim, 2*dim, 1)
        self.proj_q_5x5 = nn.Conv2d(dim, dim, 5,1,2,groups=dim)
        self.proj_k_5x5 = nn.Conv2d(dim, dim, 5,1,2,groups=dim)
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.CaPE = nn.Conv2d(dim, dim, 5,1,2,groups=dim)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x):
        """
        x: B L C
        """
        B, C, H, W = x.shape
        q, k = torch.chunk(self.proj_qk(x), 2, dim=1)
        v = self.proj_v(x)
        cape = self.CaPE(v)

        ### Img2Window

        q = self.im2cswin(self.proj_q_5x5(q)+cape)
        k = self.im2cswin(self.proj_k_5x5(k)+cape)
        # lepe = self.get_lepe(v)
        v = self.im2cswin(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W) +cape  # B C H' W'

        return x


######  Asymmetric Content-aided Window-based Multi-head Self-Attention  ######
class ACaWMSA(nn.Module):
    r""" Asymmetric Content-aided Window-based Multi-head Self-Attention.
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super(ACaWMSA, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ration = mlp_ratio
        "shift_size must in 0-window_size"
        assert 0 <= self.shift_size < self.window_size
        # self.norm1 = norm_layer(dim)
        self.attn1 = CaSA(dim//4, window_size, 3)  #16*4
        self.attn2 = CaSA(dim//4, window_size, 0)  #4*16
        self.attn3 = CaSA(dim//4, window_size, 1)  #8*8
        self.attn4 = CaSA(dim//4, window_size, 2)  #16*16
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EFFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj1 = nn.Conv2d(dim, dim, 1)
        # self.proj_conv = nn.Conv2d(dim, dim, 3, 1, padding=1, groups=dim)

    def forward(self, x):  # x: B,C,H,W

        B, C, H, W = x.shape # x: B,C,H,W
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x
        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        else:
            shift_x = x

        shift_x_norm = self.norm1(shift_x).permute(0,3,1,2).contiguous()
        x_window = self.proj1(shift_x_norm)
        x_slices = torch.chunk(x_window,4,dim=1)
        x1 = self.attn1(x_slices[0])
        x2 = self.attn2(x_slices[1]+x1)
        x3 = self.attn3(x_slices[2]+x2)
        x4 = self.attn4(x_slices[3]+x3)
        attened_x = torch.cat([x1, x2, x3, x4], dim=1)
        del x1, x2,x3,x4

        x_reverse = self.proj(attened_x.permute(0, 2, 3, 1).contiguous())
        # x_reverse = x_reverse_spa+x_reverse_ca

        # shift reverse
        if self.shift_size > 0:
            x = torch.roll(x_reverse, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = x_reverse

        if pad_r or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            shortcut = shortcut[:, :H, :W, :].contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # x: B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()
        return x  # x: B,C,H,W


class Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size):
        super().__init__()

        swin_body = []
        self.window_size = windows_size
        for i in range(depth):
            if i % 2:
                shift_size = windows_size // 2
            else:
                shift_size = 0
            self.shift_size = shift_size
            swin_body.append(ACaWMSA(c_dim, window_size=windows_size, shift_size=shift_size,
                                                  mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                                                  act_layer=nn.GELU))
        self.swin_body = nn.Sequential(*swin_body)

    def forward(self, x):
        src = x
        _, _, H, W, = x.shape
        for body in self.swin_body:
            src = body(src)
        info_mix = src

        return info_mix


######  Asymmetric Efficient Transformer Block  ######
class AET(nn.Module):
    def __init__(self, c_dim, depth, windows_size):
        super().__init__()
        modules_body = []
        modules_body.append(R_EFEM(c_dim, c_dim, 3, padding='same'))
        modules_body.extend([Spatial_Attn(c_dim, depth, windows_size)])
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


######  Residual Efficient Transformer Group  ######
class RETG(nn.Module):
    def __init__(self, c_dim, reduction, RC_depth, RS_depth, depth, windows_size):
        super(RETG, self).__init__()

        self.body_1 = []
        self.body_1.extend([AET(c_dim, depth, windows_size) for _ in range(RS_depth)])
        self.body_1.append(EFEM(c_dim, c_dim, 3, padding='same'))
        self.res_spatial_attn = nn.Sequential(*self.body_1)

        self.distill = nn.Conv2d(c_dim, c_dim // 2, 1)
        self.fusion = nn.Conv2d(int(c_dim * 1.5), c_dim, 1)

    def forward(self, x):
        short_cut = x
        d0 = self.distill(x)
        x = self.res_spatial_attn(x)
        out_B = self.fusion(torch.cat([d0, x], dim=1))
        out_lr = out_B + short_cut
        return out_lr


######  Asymmetric Content-aided Transformer ######
@ARCH_REGISTRY.register()
class ACT(nn.Module):
    def __init__(self, rgb_mean=[0.4488, 0.4371, 0.4040], upscale_factor=4, c_dim=60, reduction=16, Bsc_depth=4, RS_depth=2, RC_depth=0, depth=2,
                 windows_size=16):
        super(ACT, self).__init__()
        self.body = []
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_shallow = nn.Conv2d(3, c_dim, kernel_size=(3, 3), stride=(1, 1), padding='same')
        self.body.extend([RETG(c_dim, reduction, RC_depth, RS_depth, depth, windows_size) for _ in range(Bsc_depth)])
        self.conv_before_upsample = nn.Sequential(EFEM(c_dim, c_dim, 3, padding='same'))
        self.upsample = nn.Sequential(Pixelshuffle_Block(c_dim, 3, upscale_factor=upscale_factor, kernel_size=3))
        self.bsc_layer = nn.Sequential(*self.body)
        self.c = nn.Conv2d(Bsc_depth * c_dim, c_dim, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        out_fea = self.conv_shallow(x)
        x1 = self.bsc_layer[0](out_fea)
        x2 = self.bsc_layer[1](x1)
        x3 = self.bsc_layer[2](x2)
        x4 = self.bsc_layer[3](x3)
        out_B = self.c(torch.cat([x1, x2, x3, x4], dim=1))

        out_lr = self.conv_before_upsample(out_B) + out_fea

        output = self.upsample(out_lr) + self.mean

        return output


if __name__ == '__main__':

    model = ACT(rgb_mean=[0.4488, 0.4371, 0.4040], upscale_factor=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, 256, 256))
    model.cuda()
    out = model(x.cuda())
    print(out.shape)