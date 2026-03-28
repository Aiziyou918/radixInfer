from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from radixinfer.distributed import DistributedCommunicator, get_tp_info
from radixinfer.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """TP-aware linear layer base — holds local shard of weight."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """Full weight replicated on every TP rank."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        super().__init__(input_size, output_size, input_size, output_size, has_bias)


class LinearColParallelMerged(_LinearTPImpl):
    """Column-parallel linear for fused projections (gate+up, etc.).

    Each rank holds 1/tp_size of each output segment.
    """

    def __init__(self, input_size: int, output_sizes: List[int], has_bias: bool):
        tp_info = get_tp_info()
        local_out_sizes = [div_even(s, tp_info.size) for s in output_sizes]
        full_osize = sum(output_sizes)
        local_osize = sum(local_out_sizes)
        super().__init__(input_size, full_osize, input_size, local_osize, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """Fused QKV projection with head-level TP sharding."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(hidden_size, full_osize, hidden_size, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """Output projection (attention); input is sharded, all_reduce after matmul."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        local_isize = div_even(input_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_isize, output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """Row-parallel linear (down_proj, etc.); input sharded, all_reduce after."""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        local_isize = div_even(input_size, tp_info.size)
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_isize, output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
