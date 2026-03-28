from __future__ import annotations

from .base import DecodeInput, DecodeOutput


class DummyEngine:
    def decode(self, batch: DecodeInput) -> DecodeOutput:
        outputs = []
        for request_id, token_ids in zip(batch.request_ids, batch.token_ids, strict=True):
            last = token_ids[-1] if token_ids else request_id % 97
            outputs.append(((last + request_id + 1) % 95) + 1)
        return DecodeOutput(next_token_ids=outputs)
