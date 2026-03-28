from multiprocessing import Queue

from radixinfer.config import ServerConfig
from radixinfer.runtime.scheduler import SchedulerRuntime
from radixinfer.runtime.types import RuntimeRequest
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


def test_matched_prefix_is_copied_into_request_cache_reservation() -> None:
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
    assert runtime.page_pool.read_span(req2.cache_span) == [1, 2, 3, 4, 5, 6]


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
    assert seen["token_ids"] == [[7, 8, 9, 10]]
    assert seen["kv_shapes"] == [((3, 4, 2, 8), (3, 4, 2, 8))]
