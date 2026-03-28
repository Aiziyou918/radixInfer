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
