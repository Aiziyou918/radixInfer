from multiprocessing import Queue

from radixinfer.config import ServerConfig
from radixinfer.runtime.scheduler import SchedulerRuntime
from radixinfer.runtime.types import RequestPhase, RuntimeRequest
from radixinfer.transport.protocol import SamplingParams


def test_chunked_prefill_requires_multiple_ticks() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_prefill_tokens=4,
            max_batch_size=2,
            page_size=2,
            total_pages=64,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=1,
        prompt_tokens=list(range(10)),
        sampling=SamplingParams(max_tokens=4),
    )
    runtime.requests[1] = req

    runtime._run_prefill([1])
    assert req.phase == RequestPhase.PREFILLING
    assert req.prefill_cursor == 4
    assert not req.prefill_complete

    runtime._run_prefill([1])
    assert req.phase == RequestPhase.PREFILLING
    assert req.prefill_cursor == 8
    assert not req.prefill_complete

    runtime._run_prefill([1])
    assert req.phase == RequestPhase.READY_TO_DECODE
    assert req.prefill_cursor == 10
    assert req.prefill_complete


def test_decode_reports_length_finish_reason() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_batch_size=1,
            stop_token_ids=(),
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=7,
        prompt_tokens=[1, 2],
        sampling=SamplingParams(max_tokens=1),
        phase=RequestPhase.READY_TO_DECODE,
    )
    runtime.requests[7] = req
    runtime._run_decode([7])
    message = runtime.tokenizer_queue.get(timeout=1.0)
    assert message.finished is True
    assert message.finish_reason == "length"
    assert message.emit_text is True
    assert message.prompt_tokens == 2
    assert message.completion_tokens == 1


def test_decode_respects_request_eos_token() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_batch_size=1,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=3,
        prompt_tokens=[1, 2],
        sampling=SamplingParams(max_tokens=4, ignore_eos=False),
        eos_token_id=5,
        stop_token_ids=(),
        phase=RequestPhase.READY_TO_DECODE,
    )
    runtime.requests[3] = req
    runtime.engine.decode = lambda batch: type("Out", (), {"next_token_ids": [5]})()  # type: ignore[method-assign]
    runtime._run_decode([3])
    message = runtime.tokenizer_queue.get(timeout=1.0)
    assert message.finished is True
    assert message.finish_reason == "stop"
    assert message.emit_text is False


def test_decode_can_ignore_eos_when_requested() -> None:
    runtime = SchedulerRuntime(
        ServerConfig(
            model="debug",
            engine_kind="dummy",
            max_batch_size=1,
        ),
        Queue(),
        Queue(),
    )
    req = RuntimeRequest(
        request_id=4,
        prompt_tokens=[1, 2],
        sampling=SamplingParams(max_tokens=2, ignore_eos=True),
        eos_token_id=5,
        stop_token_ids=(),
        phase=RequestPhase.READY_TO_DECODE,
    )
    runtime.requests[4] = req
    runtime.engine.decode = lambda batch: type("Out", (), {"next_token_ids": [5]})()  # type: ignore[method-assign]
    runtime._run_decode([4])
    message = runtime.tokenizer_queue.get(timeout=1.0)
    assert message.finished is False
    assert message.finish_reason == "running"
    assert message.emit_text is True
