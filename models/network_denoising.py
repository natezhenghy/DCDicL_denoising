from typing import Any, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch import autograd
import models.basicblock as B
import torch.nn.functional as F
import numpy as np
from math import ceil
from .utils import *


class HeadNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], out_nc: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_nc + 1,
                      nc_x[0],
                      d_size,
                      padding=(d_size - 1) // 2,
                      bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], nc_x[0], 3, padding=1, bias=False))

        self.head_d = torch.zeros(1, out_nc, nc_x[0], d_size, d_size)

    def forward(self, y: Any, sigma: Any) -> Tuple[Tensor, Tensor]:
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        d = self.head_d.repeat(y.size(0), 1, 1, 1, 1).to(y.device)
        return x, d


class BodyNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], nc_d: List[int],
                 out_nc: int, nb: int):
        super(BodyNet, self).__init__()

        self.net_x = NetX(in_nc=in_nc, nc_x=nc_x, nb=nb)
        self.solve_fft = SolveFFT()

        self.net_d = NetD(nc_d=nc_d, out_nc=out_nc)
        self.solve_ls = SolveLS()

    def forward(self, x: Tensor, d: Tensor, y: Tensor, Y: Tensor,
                alpha_x: Tensor, beta_x: Tensor, alpha_d: float, beta_d: float,
                reg: float):
        """
            x: N, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            Y: N, C_out, 1, H, W, 2
            y: N, C_out, H, W
            alpha/beta: 1, 1, 1, 1
            reg: float
        """
        # Solve X
        X, D = self.rfft_xd(x, d)
        size_x = np.array(list(x.shape[-2:]))
        x = self.solve_fft(X, D, Y, alpha_x, size_x)
        beta_x = (1 / beta_x.sqrt()).repeat(1, 1, x.size(2), x.size(3))
        x = self.net_x(torch.cat([x, beta_x], dim=1))

        # Solve D
        if self.net_d is not None:
            d = self.solve_ls(x.unsqueeze(1), d, y.unsqueeze(2), alpha_d, reg)
            beda_d = (1 / beta_d.sqrt()).repeat(1, 1, d.size(3), d.size(4))
            size_d = [d.size(1), d.size(2)]
            d = d.view(d.size(0), d.size(1) * d.size(2), d.size(3), d.size(4))
            d = self.net_d(torch.cat([d, beda_d], dim=1))
            d = d.view(d.size(0), size_d[0], size_d[1], d.size(2), d.size(3))

        return x, d

    def rfft_xd(self, x: Tensor, d: Tensor):
        X = torch.rfft(x, 2).unsqueeze(1)
        D = p2o(d, x.shape[-2:])

        return X, D


class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 65,
                 nc_x: List[int] = [64, 128, 256, 512],
                 nb: int = 4):
        super(NetX, self).__init__()

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(in_nc, in_nc, bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(in_nc, nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], nc_x[0], bias=False, mode='C')

    def forward(self, x):
        x1 = x
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1[:, :-1, :, :])
        return x


class SolveFFT(nn.Module):
    def __init__(self):
        super(SolveFFT, self).__init__()

    def forward(self, X: Tensor, D: Tensor, Y: Tensor, alpha: Tensor,
                x_size: np.ndarray):
        """
            X: N, 1, C_in, H, W, 2
            D: N, C_out, C_in, H, W, 2
            Y: N, C_out, 1, H, W, 2
            alpha: N, 1, 1, 1
        """
        alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

        _D = cconj(D)
        Z = cmul(Y, D) + alpha * X

        factor1 = Z / alpha

        numerator = cmul(_D, Z).sum(2, keepdim=True)
        denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                           alpha.squeeze(-1)**2)
        factor2 = cmul(D, cdiv(numerator, denominator))
        X = (factor1 - factor2).mean(1)

        return torch.irfft(X, 2, signal_sizes=list(x_size))


class NetD(nn.Module):
    def __init__(self, nc_d: List[int] = [16], out_nc: int = 1):
        super(NetD, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0] + 1, out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = x
        x = self.relu(self.mlp(x))
        x = self.relu(self.mlp2(x))
        x = self.mlp3(x) + x1[:, :-1, :, :]
        return x


class CholeskySolve(autograd.Function):
    @staticmethod
    def forward(ctx, Q, P):
        L = torch.cholesky(Q)
        D = torch.cholesky_solve(P, L)  # D = Q-1 @ P
        ctx.save_for_backward(L, D)
        return D

    @staticmethod
    def backward(ctx, dLdD):
        L, D = ctx.saved_tensors
        dLdP = torch.cholesky_solve(dLdD, L)
        dLdQ = -dLdP.matmul(D.transpose(-2, -1))

        return dLdQ, dLdP


class SolveLS(nn.Module):
    def __init__(self):
        super(SolveLS, self).__init__()

        self.cholesky_solve = CholeskySolve.apply

    def forward(self, x, d, y, alpha, reg):
        """
            x: N, 1, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            y: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
            reg: float
        """
        C_in = x.shape[2]
        d_size = d.shape[-1]

        xtx_raw = self.cal_xtx(x, d_size)  # N, C_in, C_in, d_size, d_size
        xtx_unfold = F.unfold(
            xtx_raw.view(
                xtx_raw.size(0) * xtx_raw.size(1), xtx_raw.size(2),
                xtx_raw.size(3), xtx_raw.size(4)), d_size)
        xtx_unfold = xtx_unfold.view(xtx_raw.size(0), xtx_raw.size(1),
                                     xtx_unfold.size(1), xtx_unfold.size(2))

        xtx = xtx_unfold.view(xtx_unfold.size(0), xtx_unfold.size(1),
                              xtx_unfold.size(1), -1, xtx_unfold.size(3))
        xtx.copy_(xtx[:, :, :, torch.arange(xtx.size(3) - 1, -1, -1), ...])
        xtx = xtx.view(xtx.size(0), -1, xtx.size(-1))  # TODO
        index = torch.arange(
            (C_in * d_size)**2).view(C_in, C_in, d_size,
                                     d_size).permute(0, 2, 3, 1).reshape(-1)
        xtx.copy_(xtx[:, index, :])  # TODO
        xtx = xtx.view(xtx.size(0), d_size**2 * C_in, -1)

        xty = self.cal_xty(x, y, d_size)
        xty = xty.reshape(xty.size(0), xty.size(1), -1).permute(0, 2, 1)

        # reg
        alpha = alpha * x.size(3) * x.size(4) * reg / (d_size**2 * d.size(2))
        xtx[:, range(len(xtx[0])), range(len(
            xtx[0]))] = xtx[:, range(len(xtx[0])),
                            range(len(xtx[0]))] + alpha.squeeze(-1).squeeze(-1)
        xty += alpha.squeeze(-1) * d.reshape(d.size(0), d.size(1), -1).permute(
            0, 2, 1)

        # solve
        try:
            d = self.cholesky_solve(xtx, xty).view(d.size(0), C_in, d_size,
                                                   d_size, d.size(1)).permute(
                                                       0, 4, 1, 2, 3)
        except RuntimeError:
            pass

        return d

    def cal_xtx(self, x, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
        """
        padding = d_size - 1
        xtx = conv3d(x,
                     x.view(x.size(0), x.size(2), 1, 1, x.size(3), x.size(4)),
                     padding,
                     sample_wise=True)

        return xtx

    def cal_xty(self, x, y, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
            y: N, C_out, 1, H, W
        """
        padding = (d_size - 1) // 2

        xty = conv3d(x, y.unsqueeze(3), padding, sample_wise=True)
        return xty


class TailNet(nn.Module):
    def __init__(self):
        super(TailNet, self).__init__()

    def forward(self, x, d):
        y = conv2d(F.pad(x, [
            (d.size(-1) - 1) // 2,
        ] * 4, mode='circular'),
                   d,
                   sample_wise=True)

        return y


class HyPaNet(nn.Module):
    def __init__(
            self,
            in_nc: int = 1,
            nc: int = 256,
            out_nc: int = 8,
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x


class DCDicL(nn.Module):
    def __init__(self,
                 n_iter: int = 1,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256, 512],
                 out_nc: int = 1,
                 nb: int = 1,
                 d_size: int = 5,
                 **kargs):
        super(DCDicL, self).__init__()

        self.head = HeadNet(in_nc, nc_x, out_nc, d_size)

        self.body = BodyNet(in_nc=nc_x[0] + 1,
                            nc_x=nc_x,
                            nc_d=nc_x,
                            out_nc=out_nc,
                            nb=nb)
        self.tail = TailNet()

        self.hypa_list: nn.ModuleList = nn.ModuleList()
        for _ in range(n_iter):
            self.hypa_list.append(HyPaNet(in_nc=1, out_nc=4))

        self.n_iter = n_iter

    def forward(self, y, sigma):
        # padding
        h, w = y.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)
        y = F.pad(y, [0, paddingRight, 0, paddingBottom], mode='circular')

        # prepare Y
        Y = torch.rfft(y, 2)
        Y = Y.unsqueeze(2)

        # head_net
        x, d = self.head(y, sigma)

        pred = None
        preds = []
        for i in range(self.n_iter):
            hypas = self.hypa_list[i](sigma)
            alpha_x = hypas[:, 0].unsqueeze(-1)
            beta_x = hypas[:, 1].unsqueeze(-1)
            alpha_d = hypas[:, 2].unsqueeze(-1)
            beta_d = hypas[:, 3].unsqueeze(-1)

            x, d = self.body(x, d, y, Y, alpha_x, beta_x, alpha_d, beta_d,
                             0.001)

            dx = self.tail(x, d)
            dx = dx[..., :h, :w]
            pred = dx
            preds.append((pred, ))

        if self.training:
            return preds, d
        else:
            return pred, d
