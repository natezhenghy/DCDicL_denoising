import torch
import torch.nn.functional as F
from torch.nn.functional import pad


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def csum(x, y):
    # complex + real
    real = x[..., 0] + y
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)


def cabs2(x):
    return x[..., 0]**2 + x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    kernel_size = (psf.size(-2), psf.size(-1))
    psf = F.pad(psf,
                [0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])

    psf = roll(psf, kernel_size)
    psf = torch.rfft(psf, 2)

    return psf


def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,
                         int(axis_size / 2) * (-1 if not reverse else 1),
                         dims=axis)
    return psf


def conv2d(input, weight, padding=0, sample_wise=False):
    """
        sample_wise=False, normal conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (C_out, C_in, H_k, W_k)
        sample_wise=True, sample-wise conv2d:
            input - (N, C_in, H_in, W_in)
            weight - (N, C_out, C_in, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    if sample_wise:
        # input - (N, C_in, H_in, W_in) -> (1, N * C_in, H_in, W_in)
        input_sw = input.view(1,
                              input.size(0) * input.size(1), input.size(2),
                              input.size(3))

        # weight - (N, C_out, C_in, H_k, W_k) -> (N * C_out, C_in, H_k, W_k)
        weight_sw = weight.view(
            weight.size(0) * weight.size(1), weight.size(2), weight.size(3),
            weight.size(4))

        # group-wise convolution, group_size==batch_size
        out = F.conv2d(pad(input_sw, padding, mode='circular'),
                       weight_sw,
                       groups=input.size(0))
        out = out.view(input.size(0), weight.size(1), out.size(2), out.size(3))
    else:
        out = F.conv2d(pad(input, padding, mode='circular'), weight)
    return out


def conv3d(input, weight, padding=0, sample_wise=False):
    """
        sample_wise=False, normal conv3d:
            input - (N, C_in, D_in, H_in, W_in)
            weight - (C_out, C_in, D_k, H_k, W_k)
        sample_wise=True, sample-wise conv3d:
            input - (N, C_in, D_in, H_in, W_in)
            weight - (N, C_out, C_in, D_k, H_k, W_k)
    """
    if isinstance(padding, int):
        padding = [padding] * 4 + [0, 0]
    if sample_wise:
        # input - (N, C_in, D_in, H_in, W_in) -> (1, N * C_in, D_in, H_in, W_in)
        input_sw = input.view(1,
                              input.size(0) * input.size(1), input.size(2),
                              input.size(3), input.size(4))

        # weight - (N, C_out, C_in, D_k, H_k, W_k) -> (N * C_out, C_in, D_k, H_k, W_k)
        weight_sw = weight.view(
            weight.size(0) * weight.size(1), weight.size(2), weight.size(3),
            weight.size(4), weight.size(5))

        # group-wise convolution, group_size==batch_size
        out = F.conv3d(pad(input_sw, padding, mode='circular'),
                       weight_sw,
                       groups=input.size(0))
        out = out.view(input.size(0), weight.size(1), out.size(2), out.size(3),
                       out.size(4))
    else:
        out = F.conv3d(pad(input, padding, mode='circular'),
                       weight,
                       padding=padding)
    return out


def unfold5d(x, kernel_size):
    """perform 2D unfold on (the last 2 dimensions of) 5D Tensor"""
    x_reshape = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
    x_unfold = F.unfold(x_reshape, kernel_size)
    x_unfold = x_unfold.view(x.size(0), x.size(1), x_unfold.size(1),
                             x_unfold.size(2))
    return x_unfold
