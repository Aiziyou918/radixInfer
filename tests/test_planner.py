from radixinfer.runtime.planner import BatchPlanner
from radixinfer.runtime.types import RequestPhase, RuntimeRequest
from radixinfer.transport.protocol import SamplingParams


def make_request(request_id: int, phase: RequestPhase, age: int, prompt_len: int = 4) -> RuntimeRequest:
    request = RuntimeRequest(
        request_id=request_id,
        prompt_tokens=list(range(prompt_len)),
        sampling=SamplingParams(max_tokens=8),
        phase=phase,
    )
    request.age = age
    return request


def test_planner_prefers_decode_then_waiting_prefill() -> None:
    planner = BatchPlanner(max_batch_size=3, max_prefill_tokens=32)
    requests = [
        make_request(1, RequestPhase.WAIT_PREFILL, 1),
        make_request(2, RequestPhase.READY_TO_DECODE, 2),
        make_request(3, RequestPhase.DECODING, 0),
        make_request(4, RequestPhase.WAIT_PREFILL, 3),
    ]
    plan = planner.build_plan(requests)
    assert plan.decode == [2, 3]
    assert plan.prefill == [4]
