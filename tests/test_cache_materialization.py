from multiprocessing import Queue

import torch

from radixinfer.config import ServerConfig
from radixinfer.runtime.scheduler import SchedulerRuntime
from radixinfer.runtime.types import RequestPhase, RuntimeRequest
from radixinfer.transport.protocol import SamplingParams


def test_prefill_materializes_prompt_tokens_into_page_pool() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=1,
        prompt_tokens=[10, 11, 12, 13],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[1] = req
    runtime._run_prefill([1])
    assert req.prefix_span is not None
    assert runtime.page_pool.read_span(req.prefix_span) == [10, 11, 12, 13]
    assert req.table_slot is not None
    assert runtime.table_manager.token_table[req.table_slot][:4] == [10, 11, 12, 13]
    assert runtime.table_manager.page_table[req.table_slot][:4] == [0, 0, 1, 1]


def test_prefix_match_returns_cached_span_for_following_request() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req1 = RuntimeRequest(
        request_id=1,
        prompt_tokens=[1, 2, 3, 4],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[1] = req1
    runtime._run_prefill([1])

    hit = runtime.prefix_store.match([1, 2, 3, 4, 9, 10])
    assert hit.matched_tokens == 4
    assert hit.cached_span is not None
    assert runtime.page_pool.read_span(hit.cached_span) == [1, 2, 3, 4]


def test_matched_prefix_reuses_shared_pages_in_request_cache_reservation() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req1 = RuntimeRequest(
        request_id=1,
        prompt_tokens=[1, 2, 3, 4],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[1] = req1
    runtime._run_prefill([1])
    assert req1.prefix_cache_key is not None
    assert runtime.prefix_store.entry_ref_count(req1.prefix_cache_key) == 1

    req2 = RuntimeRequest(
        request_id=2,
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        sampling=SamplingParams(max_tokens=2),
        prefix_matched=4,
        prefix_span=req1.prefix_span,
    )
    runtime.requests[2] = req2
    runtime._run_prefill([2])
    assert req2.cache_span is not None
    assert req2.reservation is not None
    assert req2.reservation.shared_page_ids == req1.prefix_span.page_ids  # type: ignore[union-attr]
    assert runtime.page_pool.read_span(req2.cache_span) == [1, 2, 3, 4, 5, 6]
    for page_id in req1.prefix_span.page_ids:  # type: ignore[union-attr]
        assert runtime.page_pool.shared_refcount(page_id) >= 1


def test_decode_appends_token_into_page_backed_cache() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=1,
        prompt_tokens=[7, 8, 9, 10],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[1] = req
    runtime._run_prefill([1])
    runtime.engine.decode = lambda batch: type("Out", (), {"next_token_ids": [99]})()  # type: ignore[method-assign]
    runtime._run_decode([1])
    assert req.cache_span is not None
    assert runtime.page_pool.read_span(req.cache_span) == [7, 8, 9, 10, 99]


def test_decode_input_carries_kv_cache_view() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
            kv_cache_dim=8,
            kv_num_layers=3,
            kv_num_heads=2,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=3,
        prompt_tokens=[7, 8, 9, 10],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[3] = req
    runtime._run_prefill([3])
    seen = {}

    def fake_decode(batch):
        seen["token_ids"] = batch.token_ids
        seen["kv_shapes"] = [(tuple(view.keys.shape), tuple(view.values.shape)) for view in batch.kv_caches]
        return type("Out", (), {"next_token_ids": [77]})()

    runtime.engine.decode = fake_decode  # type: ignore[method-assign]
    runtime._run_decode([3])
    assert seen["token_ids"] == [[10]]
    assert seen["kv_shapes"] == [((3, 3, 2, 8), (3, 3, 2, 8))]


def test_executor_prepares_decode_metadata() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=8,
        prompt_tokens=[21, 22, 23, 24],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[8] = req
    runtime._run_prefill([8])
    prepared = runtime.executor.prepare_decode_batch([req])
    assert prepared.decode_input.token_ids == [[24]]
    assert prepared.metadata.positions == (3,)
    assert prepared.metadata.input_table_slots == (req.table_slot,)
    assert prepared.metadata.input_positions == (3,)
    assert prepared.metadata.write_table_slots == (req.table_slot,)
    assert prepared.metadata.write_positions == (4,)
    assert prepared.metadata.request_token_counts == (1,)
    assert prepared.metadata.request_table_states[0].table_slot == req.table_slot
    assert prepared.metadata.request_table_states[0].token_count == 4
    assert prepared.metadata.request_table_states[0].write_position == 4
    assert prepared.metadata.request_table_states[0].page_ids == (0, 0, 1, 1)
    assert prepared.metadata.request_table_states[0].token_ids == (21, 22, 23, 24)
    assert prepared.metadata.request_paged_states[0].table_slot == req.table_slot
    assert prepared.metadata.request_paged_states[0].token_count == 4
    assert prepared.metadata.request_paged_states[0].write_position == 4
    assert prepared.metadata.request_paged_states[0].page_ids == (0, 1)
    assert prepared.metadata.request_paged_states[0].page_indices == (0, 1)
    assert prepared.metadata.request_paged_states[0].kv_page_indices == (0, 1)
    assert prepared.metadata.request_paged_states[0].last_page_len == 2


def test_executor_prepares_prefill_metadata() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=9,
        prompt_tokens=[31, 32, 33, 34],
        sampling=SamplingParams(max_tokens=2),
        prefix_matched=2,
        prefill_cursor=2,
    )
    req.table_slot = runtime.table_manager.allocate()
    req.reservation = runtime.cache_manager.reserve(6, None)
    assert req.reservation is not None
    req.cache_span = runtime.page_pool.write_tokens(req.reservation, [31, 32, 33, 34])
    prepared = runtime.executor.prepare_prefill_batch([req])
    assert prepared.prefill_input.token_ids == [[33, 34]]
    assert prepared.metadata.positions == (2, 3)
    assert prepared.metadata.input_table_slots == (req.table_slot, req.table_slot)
    assert prepared.metadata.input_positions == (2, 3)
    assert prepared.metadata.write_table_slots == (req.table_slot,)
    assert prepared.metadata.write_positions == (4,)
    assert prepared.metadata.request_token_counts == (2,)
    assert prepared.metadata.request_table_states[0].table_slot == req.table_slot
    assert prepared.metadata.request_table_states[0].token_count == 2
    assert prepared.metadata.request_table_states[0].write_position == 4
    assert prepared.metadata.request_table_states[0].page_ids == (0, 0)
    assert prepared.metadata.request_table_states[0].token_ids == (31, 32)
    assert prepared.metadata.request_paged_states[0].page_ids == (0,)
    assert prepared.metadata.request_paged_states[0].last_page_len == 2


def test_executor_flattens_multi_request_prefill_metadata() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req1 = RuntimeRequest(
        request_id=10,
        prompt_tokens=[41, 42, 43, 44],
        sampling=SamplingParams(max_tokens=2),
        prefix_matched=0,
        prefill_cursor=2,
    )
    req2 = RuntimeRequest(
        request_id=11,
        prompt_tokens=[51, 52, 53, 54],
        sampling=SamplingParams(max_tokens=2),
        prefix_matched=2,
        prefill_cursor=2,
    )
    req1.table_slot = runtime.table_manager.allocate()
    req2.table_slot = runtime.table_manager.allocate()
    req1.reservation = runtime.cache_manager.reserve(6, None)
    req2.reservation = runtime.cache_manager.reserve(6, None)
    assert req1.reservation is not None
    assert req2.reservation is not None
    req1.cache_span = runtime.page_pool.write_tokens(req1.reservation, [41, 42])
    req2.cache_span = runtime.page_pool.write_tokens(req2.reservation, [51, 52, 53, 54])
    prepared = runtime.executor.prepare_prefill_batch([req1, req2])
    assert prepared.prefill_input.token_ids == [[41, 42], [53, 54]]
    assert prepared.metadata.positions == (0, 1, 2, 3)
    assert prepared.metadata.input_table_slots == (
        req1.table_slot,
        req1.table_slot,
        req2.table_slot,
        req2.table_slot,
    )
    assert prepared.metadata.input_positions == (0, 1, 2, 3)
    assert prepared.metadata.write_table_slots == (req1.table_slot, req2.table_slot)
    assert prepared.metadata.write_positions == (2, 4)
    assert prepared.metadata.request_token_counts == (2, 2)
    assert prepared.metadata.request_view(0).positions == (0, 1)
    assert prepared.metadata.request_view(1).positions == (2, 3)
    assert prepared.metadata.request_view(0).request_table_states[0].token_ids == ()
    assert prepared.metadata.request_view(1).request_table_states[0].token_ids == (51, 52)
    assert prepared.metadata.request_view(0).request_paged_states[0].page_ids == ()
    assert prepared.metadata.request_view(1).request_paged_states[0].page_ids == (3,)


def test_prefill_writes_real_kv_into_page_pool_for_hf_debug_engine() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="hf",
            device="cpu",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
            kv_cache_dim=16,
            kv_num_layers=2,
            kv_num_heads=4,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=4,
        prompt_tokens=[7, 8, 9, 10],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[4] = req
    runtime._run_prefill([4])
    assert req.cache_span is not None
    kv = runtime.page_pool.read_kv(req.cache_span)
    assert kv.token_count == 4
    assert torch.count_nonzero(kv.keys).item() > 0
    assert torch.count_nonzero(kv.values).item() > 0


def test_finished_request_unlocks_prefix_entry_but_keeps_shared_cache_resident() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=5,
        prompt_tokens=[11, 12, 13, 14],
        sampling=SamplingParams(max_tokens=1),
    )
    runtime.requests[5] = req
    runtime._run_prefill([5])
    assert req.prefix_cache_key is not None
    runtime.engine.decode = lambda batch: type("Out", (), {"next_token_ids": [99]})()  # type: ignore[method-assign]
    runtime._run_decode([5])
    assert runtime.prefix_store.entry_ref_count(req.prefix_cache_key) == 0
    assert req.prefix_span is not None
    assert runtime.page_pool.read_span(req.prefix_span) == [11, 12, 13, 14]


def test_full_prefix_hit_becomes_decode_ready_without_copying_prefix() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=8,
            max_batch_size=2,
            page_size=2,
            total_pages=16,
        ),
        Queue(),
        Queue(),
    )
    req1 = RuntimeRequest(
        request_id=6,
        prompt_tokens=[1, 2, 3, 4],
        sampling=SamplingParams(max_tokens=2),
    )
    runtime.requests[6] = req1
    runtime._run_prefill([6])

    req2 = RuntimeRequest(
        request_id=7,
        prompt_tokens=[1, 2, 3, 4],
        sampling=SamplingParams(max_tokens=2),
        prefix_matched=4,
        prefix_span=req1.prefix_span,
        prefix_cache_key=req1.prefix_cache_key,
    )
    runtime.prefix_store.lock(req2.prefix_cache_key)
    runtime.requests[7] = req2
    runtime._run_prefill([7])
    assert req2.phase == RequestPhase.READY_TO_DECODE
    assert req2.cache_span == req1.prefix_span
