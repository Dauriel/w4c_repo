from pathlib import Path
from functools import partial
from collections import namedtuple
from contextlib import contextmanager

import torch
from torch import nn, Tensor, einsum
import torch.distributed as dist
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from beartype import beartype
from beartype.typing import Tuple, Union, List, Optional, Dict, Literal

import pickle

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def safe_div(num, den, eps = 1e-10):
    return num / den.clamp(min = eps)

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

# prepare batch norm in maxvit for distributed training

def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# multi-headed rms normalization, for query / key normalized attention

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        heads
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    batchnorm_klass = MaybeSyncBatchnorm2d()

    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        batchnorm_klass(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        batchnorm_klass(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

class Attention(Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        heads = 32,
        dim_head = 32,
        dropout = 0.,
        window_size = 8,
        num_registers = 1
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        dim_inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

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

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.q_norm = RMSNorm(dim_head, heads = heads)
        self.k_norm = RMSNorm(dim_head, heads = heads)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None
    ):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # conditioning

        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q, k = self.q_norm(q), self.k_norm(k)

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MaxViT(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        cond_dim = 32,   # for conditioniong on lead time embedding
        heads = 32,
        dim_head = 32,
        window_size = 8,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        num_register_tokens = 4
    ):
        super().__init__()
        depth = (depth,) if isinstance(depth, int) else depth
        assert num_register_tokens > 0

        self.cond_dim = cond_dim

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                block_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                grid_attn = Attention(dim = layer_dim, cond_dim = cond_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([
                    conv,
                    block_attn,
                    grid_attn
                ]))

                self.register_tokens.append(register_tokens)

    def forward(
        self,
        x: Tensor,
        cond: Tensor
    ):

        b, w = x.shape[0], self.window_size

        for (conv, block_attn, grid_attn), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = block_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = grid_attn(x, cond = cond) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        return x
      
