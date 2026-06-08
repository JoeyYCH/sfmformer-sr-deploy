"""
IDynamic / IDynamicDWConv -- dynamic depth-wise convolution.

Original (training) implementation used a CuPy-compiled CUDA kernel:
    https://github.com/Atten4Vis/DemystifyLocalViT
    Han, Fan, Dai, Sun, Cheng, Liu, Wang.
    "On the Connection between Local Attention and Dynamic Depth-wise
    Convolution", ICLR 2022 (Spotlight).

This DEPLOYMENT version replaces that CUDA kernel with a pure-PyTorch
implementation (im2col / F.unfold). It is numerically identical to the
original kernel (verified to ~1e-6), but:
  * needs no CuPy and compiles no CUDA at runtime, and
  * uses one code path for CPU and GPU,
so the model loads and runs unchanged on a Raspberry Pi (CPU) and on a
Windows / Linux machine with an NVIDIA GPU. The class definitions below are
byte-for-byte the same layers as the original, so existing checkpoints load
with strict=True.

@inproceedings{han2021connection,
  title={On the Connection between Local Attention and Dynamic Depth-wise Convolution},
  author={Han, Qi and Fan, Zejia and Dai, Qi and Sun, Lei and Cheng, Ming-Ming and Liu, Jiaying and Wang, Jingdong},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """Dynamic depth-wise convolution with per-pixel, per-group kernels.

    input : (N, C, H, W)
    weight: (N, G, KH, KW, Hout, Wout)   one KHxKW kernel per (group, pixel)
    return: (N, C, Hout, Wout)

    For output (n, c, h, w) with group g = c // (C // G):
        out[n,c,h,w] = sum_{kh,kw} weight[n,g,kh,kw,h,w]
                                   * input[n,c, h*s-pad+kh*dil, w*s-pad+kw*dil]
    (out-of-range taps treated as zero). This is exactly the original CuPy
    forward kernel, expressed with F.unfold so it runs on any device and
    differentiates natively (no custom autograd Function needed).
    """
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    N, C, H, W = input.shape
    G, KH, KW = weight.shape[1], weight.shape[2], weight.shape[3]
    Hout, Wout = weight.shape[4], weight.shape[5]

    # (N, C*KH*KW, L); L = Hout*Wout, columns in row-major output order
    cols = F.unfold(input, (KH, KW), dilation=dilation,
                    padding=padding, stride=stride)
    L = cols.shape[-1]
    cpg = C // G                                         # channels per group
    cols = cols.view(N, G, cpg, KH * KW, L)              # (N,G,cpg,KHKW,L)
    w = weight.reshape(N, G, KH * KW, L).unsqueeze(2)    # (N,G,1,KHKW,L)
    out = (cols * w).sum(dim=3).reshape(N, C, Hout, Wout)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


class IDynamicDWConv(nn.Module):
    """HyperNet that predicts per-pixel dynamic depth-wise kernels."""

    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super(IDynamicDWConv, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=channels // reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio,
                      kernel_size ** 2 * self.groups, 1, bias=bias)
        )

    def forward(self, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size,
                             self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1,
                             padding=(self.kernel_size - 1) // 2)
        return out


class IDynamic(nn.Module):
    """Two-input variant: HyperNet reads `x`, the filter is applied to `x_main`."""

    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super(IDynamic, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 8
        self.group_channels = group_channels
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=channels // reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio,
                      kernel_size ** 2 * self.groups, 1, bias=bias)
        )

    def forward(self, x_main, x):
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size,
                             self.kernel_size, h, w)
        out = _idynamic_cuda(x_main, weight, stride=1,
                             padding=(self.kernel_size - 1) // 2)
        return out