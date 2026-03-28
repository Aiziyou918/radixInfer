from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self._rmsnorm = None

    def _get_rmsnorm(self):
        if self._rmsnorm is None:
            try:
                from flashinfer import rmsnorm
                self._rmsnorm = rmsnorm
            except ImportError:
                self._rmsnorm = self._torch_rmsnorm
        return self._rmsnorm

    def _torch_rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float, out=None):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        y = weight.float() * x
        y = y.to(orig_dtype)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_rmsnorm()(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self._get_rmsnorm()(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """Fused add + RMSNorm for residual connections."""

    def __init__(self, size: int, eps: float) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self._rmsnorm = None
        self._fused_add_rmsnorm = None

    def _get_ops(self):
        if self._rmsnorm is None:
            try:
                from flashinfer import fused_add_rmsnorm, rmsnorm
                self._rmsnorm = rmsnorm
                self._fused_add_rmsnorm = fused_add_rmsnorm
            except ImportError:
                self._rmsnorm = RMSNorm(self.weight.shape[0], self.eps)._torch_rmsnorm
                self._fused_add_rmsnorm = self._torch_fused_add_rmsnorm
        return self._rmsnorm, self._fused_add_rmsnorm

    def _torch_fused_add_rmsnorm(self, x, residual, weight, eps):
        x.add_(residual)
        residual.copy_(x)
        orig_dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + eps)
        x.copy_((weight.float() * xf).to(orig_dtype))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rmsnorm, fused_add_rmsnorm = self._get_ops()
        if residual is None:
            return rmsnorm(x, self.weight, self.eps), x
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
