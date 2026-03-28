from __future__ import annotations

from dataclasses import dataclass

from .types import RequestPhase, RuntimeRequest


@dataclass(frozen=True)
class Plan:
    prefill: list[int]
    decode: list[int]


class BatchPlanner:
    def __init__(self, max_batch_size: int, max_prefill_tokens: int) -> None:
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens

    def build_plan(self, requests: list[RuntimeRequest]) -> Plan:
        active = [req for req in requests if not req.finished]
        decode_ready = [
            req for req in active if req.phase in {RequestPhase.READY_TO_DECODE, RequestPhase.DECODING}
        ]
        decode_ready.sort(key=lambda req: (-req.age, req.request_id))

        decode_ids = [req.request_id for req in decode_ready[: self.max_batch_size]]
        remaining_slots = self.max_batch_size - len(decode_ids)
        if remaining_slots <= 0:
            return Plan(prefill=[], decode=decode_ids)

        waiting = [req for req in active if req.phase == RequestPhase.WAIT_PREFILL]
        waiting.sort(key=lambda req: (-req.age, req.request_id))

        prefill_ids: list[int] = []
        token_budget = self.max_prefill_tokens
        for req in waiting:
            cost = max(1, len(req.prompt_tokens) - req.prefix_matched)
            if cost > token_budget and prefill_ids:
                break
            prefill_ids.append(req.request_id)
            token_budget -= min(cost, token_budget)
            if len(prefill_ids) >= remaining_slots:
                break
        return Plan(prefill=prefill_ids, decode=decode_ids)
