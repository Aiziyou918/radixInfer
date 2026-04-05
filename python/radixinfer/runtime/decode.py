from __future__ import annotations

from typing import Iterable

from radixinfer.core import Batch, Req


class DecodeManager:
    """Tracks in-flight decode requests keyed by uid.

    Using dict[uid, Req] instead of Set[Req] so that abort_req and existence
    checks are O(1).  The sorted() in schedule_next_batch is the only O(n log n)
    operation, and it is necessary for TP-rank determinism regardless of the
    underlying container.
    """

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self._reqs: dict[int, Req] = {}

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        for req in reqs:
            self._reqs[req.uid] = req
        finished = [uid for uid, r in self._reqs.items() if not r.can_decode]
        for uid in finished:
            del self._reqs[uid]

    def remove_req(self, req: Req) -> None:
        self._reqs.pop(req.uid, None)

    def abort_req(self, uid: int) -> Req | None:
        return self._reqs.pop(uid, None)

    @property
    def inflight_tokens(self) -> int:
        page_overhead = (self.page_size - 1) * len(self._reqs)
        return sum(r.remain_len for r in self._reqs.values()) + page_overhead

    def schedule_next_batch(self) -> Batch | None:
        if not self._reqs:
            return None
        # Sort by uid for deterministic ordering across TP ranks.
        # Without sorting, dict iteration order can differ between processes,
        # causing token positions to diverge → allreduce mismatch → NCCL deadlock.
        reqs = sorted(self._reqs.values(), key=lambda r: r.uid)
        return Batch(reqs=reqs, phase="decode")

    @property
    def runnable(self) -> bool:
        return bool(self._reqs)
