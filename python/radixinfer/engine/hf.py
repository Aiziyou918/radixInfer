from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from .base import DecodeInput, DecodeOutput


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class HuggingFaceEngine:
    model_name: str
    device: str = "auto"

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)
        self.model = self._load_model().eval().to(self.device)

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
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(tokens, dtype=torch.long) for tokens in batch.token_ids],
            batch_first=True,
            padding_value=0,
        ).to(self.device)
        attention_mask = (padded != 0).long()
        logits = self.model(input_ids=padded, attention_mask=attention_mask).logits
        if batch.kv_caches:
            for row, kv_cache in enumerate(batch.kv_caches):
                kv_bias_token = int(torch.sum(kv_cache.values).item()) % logits.shape[-1]
                logits[row, -1, kv_bias_token] += 1e6
        next_token_ids = logits[:, -1, :].argmax(dim=-1).tolist()
        return DecodeOutput(next_token_ids=next_token_ids)
