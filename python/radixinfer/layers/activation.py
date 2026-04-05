from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _fallback_act_and_mul(
    x: torch.Tensor, out: torch.Tensor | None, act_fn
) -> torch.Tensor:
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    result = act_fn(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    try:
        from flashinfer import silu_and_mul as _f
        return _f(x, out=out)
    except ImportError:
        return _fallback_act_and_mul(x, out, torch.nn.functional.silu)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    try:
        from flashinfer import gelu_and_mul as _f
        return _f(x, out=out)
    except ImportError:
        return _fallback_act_and_mul(x, out, torch.nn.functional.gelu)


__all__ = ["silu_and_mul", "gelu_and_mul"]
