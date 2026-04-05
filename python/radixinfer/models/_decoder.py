from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from radixinfer.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from radixinfer.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP, RopeAttn

if TYPE_CHECKING:
    from .config import ModelConfig


def build_decoder_model(attn_kwargs: dict | None = None) -> type[BaseLLMModel]:
    """Return a ForCausalLM class using RopeAttn + GatedMLP decoder layers.

    All standard Llama-style architectures share the same structure; the only
    variation is keyword arguments forwarded to RopeAttn (e.g. has_qk_norm,
    has_attn_bias).  Pass them via *attn_kwargs*.
    """
    _attn_kwargs: dict = attn_kwargs or {}

    class DecoderLayer(BaseOP):
        def __init__(self, config: ModelConfig, layer_id: int):
            self.self_attn = RopeAttn(config, layer_id, **_attn_kwargs)
            self.mlp = GatedMLP(config)
            self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
            self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
            self._layer_id = layer_id

        @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
        def forward(
            self,
            x: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            x, residual = self.input_layernorm.forward(x, residual)
            x = self.self_attn.forward(x)
            x, residual = self.post_attention_layernorm.forward(x, residual)
            x = self.mlp.forward(x)
            return x, residual

    class DecoderModel(BaseOP):
        def __init__(self, config: ModelConfig):
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            self.layers = OPList(
                [DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
            )
            self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embed_tokens.forward(input_ids)
            residual: torch.Tensor | None = None
            for layer in self.layers.op_list:
                x, residual = layer.forward(x, residual)
            return self.norm.forward(x, residual)[0]

    class ForCausalLM(BaseLLMModel):
        def __init__(self, config: ModelConfig):
            self.model = DecoderModel(config)
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )

        def forward(self) -> torch.Tensor:
            from radixinfer.core import get_global_ctx

            output = self.model.forward(get_global_ctx().batch.input_ids)
            return self.lm_head.forward(output)

    return ForCausalLM
