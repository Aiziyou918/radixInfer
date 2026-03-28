from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from radixinfer.distributed import DistributedCommunicator, get_tp_info
from radixinfer.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """Vocabulary embedding sharded along vocab dimension across TP ranks."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_info.rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from radixinfer.kernel import indexing

            y = indexing(
                weights=self.weight,
                indices=x,
                vocab_range=self.vocab_range if self.tp_size > 1 else None,
            )
        except ImportError:
            # fallback when kernel extension is not compiled
            if self.tp_size > 1:
                mask = (x >= self.vocab_range[0]) & (x < self.vocab_range[0] + self.vocab_range[1])
                local_x = (x - self.vocab_range[0]).clamp(min=0)
                y = F.embedding(local_x, self.weight) * mask.unsqueeze(-1).to(self.weight.dtype)
            else:
                y = F.embedding(x, self.weight)

        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """LM head (weight-tied or independent) with optional TP all_gather."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        # When tied, pop any spurious lm_head weights from checkpoint
        for key in [f"{prefix}.weight", f"{prefix}.bias"]:
            state_dict.pop(key, None)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from radixinfer.core import get_global_ctx

        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits

        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)
        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
