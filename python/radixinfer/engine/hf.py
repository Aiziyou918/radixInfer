from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from .attention import HuggingFaceAttentionBackend
from .base import DecodeInput, DecodeOutput, PrefillInput, PrefillOutput


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class HuggingFaceEngine:
    model_name: str
    device: str = "auto"
    kv_num_layers: int = 2
    kv_num_heads: int = 2
    kv_cache_dim: int = 16

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)
        self.model = self._load_model().eval().to(self.device)
        num_layers = getattr(self.model.config, "n_layer", None) or getattr(
            self.model.config, "num_hidden_layers", self.kv_num_layers
        )
        num_heads = getattr(self.model.config, "n_head", None) or getattr(
            self.model.config, "num_attention_heads", self.kv_num_heads
        )
        hidden_size = getattr(self.model.config, "n_embd", None) or getattr(
            self.model.config, "hidden_size", num_heads * self.kv_cache_dim
        )
        head_dim = hidden_size // max(1, num_heads)
        self.attention = HuggingFaceAttentionBackend(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=self.model.dtype,
        )

    def _load_model(self):
        if self.model_name == "debug":
            config = GPT2Config(
                vocab_size=512,
                n_positions=256,
                n_ctx=256,
                n_embd=64,
                n_layer=2,
                n_head=4,
                bos_token_id=1,
                eos_token_id=2,
            )
            return GPT2LMHeadModel(config)
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    @torch.inference_mode()
    def prefill(self, batch: PrefillInput) -> PrefillOutput:
        prepared = self.attention.prepare_batch(batch.token_ids, batch.kv_caches, batch.metadata)
        outputs = [
            self.model(
                input_ids=inputs.input_ids,
                past_key_values=inputs.past_key_values,
                use_cache=True,
            )
            for inputs in prepared
        ]
        kv_writes = self.attention.extract_cache_writes(
            outputs,
            [len(token_ids) for token_ids in batch.token_ids],
        )
        return PrefillOutput(kv_writes=kv_writes)

    @torch.inference_mode()
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        prepared = self.attention.prepare_batch(batch.token_ids, batch.kv_caches, batch.metadata)
        outputs = [
            self.model(
                input_ids=inputs.input_ids,
                past_key_values=inputs.past_key_values,
                use_cache=True,
            )
            for inputs in prepared
        ]
        next_token_ids = [int(output.logits[:, -1, :].argmax(dim=-1).item()) for output in outputs]
        kv_writes = self.attention.extract_cache_writes(
            outputs,
            [len(token_ids) for token_ids in batch.token_ids],
        )
        return DecodeOutput(next_token_ids=next_token_ids, kv_writes=kv_writes)
