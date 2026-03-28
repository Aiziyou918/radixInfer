from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    try:
        from flashinfer import silu_and_mul as _silu_and_mul
        return _silu_and_mul(x, out=out)
    except ImportError:
        import torch as _torch
        half = x.shape[-1] // 2
        gate, up = x[..., :half], x[..., half:]
        result = _torch.nn.functional.silu(gate) * up
        if out is not None:
            out.copy_(result)
            return out
        return result


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    try:
        from flashinfer import gelu_and_mul as _gelu_and_mul
        return _gelu_and_mul(x, out=out)
    except ImportError:
        import torch as _torch
        half = x.shape[-1] // 2
        gate, up = x[..., :half], x[..., half:]
        result = _torch.nn.functional.gelu(gate) * up
        if out is not None:
            out.copy_(result)
            return out
        return result


__all__ = ["silu_and_mul", "gelu_and_mul"]
