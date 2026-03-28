from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import StateLessOP

if TYPE_CHECKING:
    pass


class MoELayer(StateLessOP):
    """Mixture-of-Experts layer stub — delegates to moe backend."""

    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        from radixinfer.core import get_global_ctx

        ctx = get_global_ctx()
        if not hasattr(ctx, "moe_backend") or ctx.moe_backend is None:
            raise RuntimeError("MoELayer requires a moe_backend set in the global Context")
        return ctx.moe_backend.forward(x, router_logits, self.layer_id)
