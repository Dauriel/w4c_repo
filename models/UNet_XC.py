import torch
from torch import nn, Tensor, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from beartype import beartype
from beartype.typing import Tuple, Union, List, Optional, Dict, Literal

from models.unet_parts import OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS

def exists(val):
    return val is not None

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

class XCAttention(Module):
    """
    this specific linear attention was proposed in https://arxiv.org/abs/2106.09681 (El-Nouby et al.)
    """
    @beartype
    def __init__(
        self,
        *,
        dim,
        cond_dim: Optional[int] = None,
        dim_head = 32,
        heads = 8,
        scale = 8,
        flash = False,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.has_cond = exists(cond_dim)

        self.film = None

        if self.has_cond:
            self.film = Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange('b (r d) -> r b 1 d', r = 2)
            )

        self.norm = nn.LayerNorm(dim, elementwise_affine = not self.has_cond)

        self.to_qkv = Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv = 3, h = heads)
        )

        self.scale = scale

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attn_dropout = nn.Dropout(dropout)

        self.to_out = Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(
        self,
        x,
        cond: Optional[Tensor] = None
    ):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack_one(x, 'b * c')

        x = self.norm(x)

        # conditioning

        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # cosine sim linear attention

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        sim = einsum('b h i n, b h j n -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j n -> b h i n', attn, v)

        out = self.to_out(out)

        out = unpack_one(out, ps, 'b * c')
        return rearrange(out, 'b h w c -> b c h w')

class SmaAt_UNet(nn.Module):
    def __init__(self, in_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.xc_att1 = XCAttention(dim=64, heads=8, scale=8)  # XCAttention instead of CBAM

        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.xc_att2 = XCAttention(dim=128, heads=8, scale=8)  # XCAttention instead of CBAM

        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.xc_att3 = XCAttention(dim=256, heads=8, scale=8)  # XCAttention instead of CBAM

        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.xc_att4 = XCAttention(dim=512, heads=8, scale=8)  # XCAttention instead of CBAM

        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.xc_att5 = XCAttention(dim=1024 // factor, heads=8, scale=8)  # XCAttention instead of CBAM

        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1AttXC = self.xc_att1(x1)

        x2 = self.down1(x1AttXC)
        x2AttXC = self.xc_att2(x2)

        x3 = self.down2(x2AttXC)
        x3AttXC = self.xc_att3(x3)

        x4 = self.down3(x3AttXC)
        x4AttXC = self.xc_att4(x4)

        x5 = self.down4(x4AttXC)
        x5AttXC = self.xc_att5(x5)

        x = self.up1(x5AttXC, x4AttXC)
        x = self.up2(x, x3AttXC)
        x = self.up3(x, x2AttXC)
        x = self.up4(x, x1AttXC)

        logits = self.outc(x)
        return logits
