from collections import OrderedDict
from typing import Any
import torch.nn as nn
"""
--------------------------------------------
Kai Zhang, https://github.com/cszn/KAIR
--------------------------------------------
https://github.com/xinntao/BasicSR
--------------------------------------------
"""


def sequential(*args: Any):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64,
         out_channels=64,
         kernel_size=3,
         stride=1,
         padding=1,
         bias=True,
         mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias))
        elif t == 'T':
            L.append(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   bias=bias))
        elif t == 'B':
            L.append(
                nn.BatchNorm2d(out_channels,
                               momentum=0.9,
                               eps=1e-04,
                               affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope,
                                  inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                             padding=0))
        elif t == 'A':
            L.append(
                nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                             padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride,
                        padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


def upsample_convtranspose(in_channels=64,
                           out_channels=3,
                           kernel_size=2,
                           stride=2,
                           padding=0,
                           bias=True,
                           mode='2R',
                           negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in [
        '2', '3', '4'
    ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,
               mode, negative_slope)
    return up1


def downsample_strideconv(in_channels=64,
                          out_channels=64,
                          kernel_size=2,
                          stride=2,
                          padding=0,
                          bias=True,
                          mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in [
        '2', '3', '4'
    ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias,
                 mode, negative_slope)
    return down1
