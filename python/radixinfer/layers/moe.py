from __future__ import annotations

import torch

from .base import BaseOP


class MoELayer(BaseOP):
    """Mixture-of-Experts layer with per-expert weight tensors."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()
        from radixinfer.distributed import DistributedCommunicator, get_tp_info
        from radixinfer.utils import div_even

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input

        self._comm = DistributedCommunicator()
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        inter_per_rank = div_even(intermediate_size, tp_info.size)

        # Weight tensors discovered by BaseOP.state_dict() / load_state_dict()
        self.gate_up_proj = torch.empty(num_experts, 2 * inter_per_rank, hidden_size)
        self.down_proj = torch.empty(num_experts, hidden_size, inter_per_rank)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        from radixinfer.core import get_global_ctx

        ctx = get_global_ctx()
        if ctx.moe_backend is None:
            raise RuntimeError("MoELayer requires a moe_backend set in the global Context")
        result = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            result = self._comm.all_reduce(result)
        return result
