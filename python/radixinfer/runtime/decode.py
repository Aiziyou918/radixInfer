from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from radixinfer.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """Merge new reqs into running set, dropping finished ones."""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """Estimate tokens reserved by in-flight decode requests (for prefill budgeting)."""
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # Sort by uid for deterministic ordering across TP ranks.
        # running_reqs is a set; without sorting, list() produces different
        # orderings on rank 0 and rank 1 (different Python object ids), causing
        # token positions to diverge, allreduce to mix incompatible tensors, and
        # eventually NCCL seqnum desync under sustained load.
        reqs = sorted(self.running_reqs, key=lambda r: r.uid)
        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
