from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch

from radixinfer.core import Batch, Req

if TYPE_CHECKING:
    from radixinfer.cache.kv_pool import BaseCacheHandle  # type: ignore[attr-defined]

    from radixinfer.runtime.cache_manager import CacheManager
    from radixinfer.runtime.decode import DecodeManager
    from radixinfer.runtime.table import TableManager
    from radixinfer.runtime.utils import PendingReq
    from radixinfer.transport.protocol import TokenizedRequest


class ChunkedReq(Req):
    """A prefill request that spans multiple scheduling ticks (chunked prefill)."""

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(
        self, req: PendingReq
    ) -> Tuple[object, int] | None:  # (cache_handle, table_idx)
        if self.table_manager.available_size == 0:
            return None

        match_result = self.cache_manager.match_req(req)
        handle = match_result.cuda_handle
        cached_len = handle.cached_len
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None

        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        table_idx = self.table_manager.allocate()
        if cached_len > 0:
            # Copy cached token ids and page indices into table
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: object,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        from radixinfer.core import SamplingParams

        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len

        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        # Continue an in-progress chunked request
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # Try to allocate a new request
        resource = self._try_allocate_one(pending_req)
        if resource is not None:
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: TokenizedRequest) -> None:
        from radixinfer.core import SamplingParams
        from radixinfer.runtime.utils import PendingReq as PR

        sampling = req.sampling
        stop_ids: list[int] = list(getattr(req, "stop_token_ids", ()))
        eos = getattr(req, "eos_token_id", None)
        if eos is not None and eos not in stop_ids:
            stop_ids.append(eos)

        sp = SamplingParams(
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            ignore_eos=sampling.ignore_eos,
            max_tokens=sampling.max_tokens,
            stop_token_ids=stop_ids,
        )
        input_ids = torch.tensor(req.token_ids, dtype=torch.int32)
        self.pending_list.append(PR(uid=req.request_id, input_ids=input_ids, sampling_params=sp))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if not self.pending_list:
            return None

        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []

        for pending_req in self.pending_list:
            req = adder.try_add_one(pending_req)
            if req is not None:
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break

        if not reqs:
            return None

        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
