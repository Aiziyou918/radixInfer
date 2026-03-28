from __future__ import annotations

import torch

from .base import DecodeInput, DecodeOutput


class DummyEngine:
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        outputs = []
        kv_caches = batch.kv_caches or [None] * len(batch.request_ids)
        for request_id, token_ids, kv_cache in zip(
            batch.request_ids, batch.token_ids, kv_caches, strict=True
        ):
            last = token_ids[-1] if token_ids else request_id % 97
            kv_bias = 0
            if kv_cache is not None:
                kv_bias = int(torch.sum(kv_cache.keys).item()) % 17
            outputs.append(((last + request_id + kv_bias + 1) % 95) + 1)
        return DecodeOutput(next_token_ids=outputs)
