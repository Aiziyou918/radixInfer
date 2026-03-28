from __future__ import annotations

import torch

from .base import AttentionCacheWrite, DecodeInput, DecodeOutput, PrefillInput, PrefillOutput


class DummyEngine:
    def prefill(self, batch: PrefillInput) -> PrefillOutput:
        return PrefillOutput(
            kv_writes=[self._make_kv_write(token_ids) for token_ids in batch.token_ids]
        )

    def decode(self, batch: DecodeInput) -> DecodeOutput:
        outputs = []
        kv_writes = []
        kv_caches = batch.kv_caches or [None] * len(batch.request_ids)
        for request_id, token_ids, kv_cache in zip(
            batch.request_ids, batch.token_ids, kv_caches, strict=True
        ):
            last = token_ids[-1] if token_ids else request_id % 97
            kv_bias = 0
            if kv_cache is not None:
                kv_bias = int(torch.sum(kv_cache.keys).item()) % 17
            outputs.append(((last + request_id + kv_bias + 1) % 95) + 1)
            kv_writes.append(self._make_kv_write(token_ids))
        return DecodeOutput(next_token_ids=outputs, kv_writes=kv_writes)

    def _make_kv_write(self, token_ids: list[int]) -> AttentionCacheWrite:
        if not token_ids:
            empty = torch.empty((0, 0, 0, 0), dtype=torch.float32)
            return AttentionCacheWrite(keys=empty, values=empty, token_count=0)
        values = torch.tensor(token_ids, dtype=torch.float32).view(1, len(token_ids), 1, 1)
        keys = values + 1.0
        return AttentionCacheWrite(keys=keys, values=values, token_count=len(token_ids))
