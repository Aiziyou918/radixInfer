from multiprocessing import Queue

from radixinfer.transport.protocol import DetokenizeRequest, SamplingParams, TokenizeRequest
from radixinfer.transport.tokenizer_worker import (
    SimpleTokenizer,
    TokenizerProcess,
    create_tokenizer_backend,
)


def test_tokenizer_backend_falls_back_to_simple_for_debug() -> None:
    tokenizer = create_tokenizer_backend("debug")
    assert isinstance(tokenizer, SimpleTokenizer)


def test_simple_tokenizer_exposes_stop_tokens() -> None:
    tokenizer = SimpleTokenizer()
    assert tokenizer.eos_token_id == 0
    assert tokenizer.stop_token_ids == (0,)


def test_tokenizer_process_emits_tokenizer_metadata() -> None:
    ingress = Queue()
    runtime_queue = Queue()
    frontend_queue = Queue()
    worker = TokenizerProcess(ingress, runtime_queue, frontend_queue, "debug")

    ingress.put(TokenizeRequest(request_id=1, prompt="abc", sampling=SamplingParams(max_tokens=2)))
    ingress.put(None)
    worker.run()

    tokenized = runtime_queue.get(timeout=1.0)
    assert tokenized.eos_token_id == 0
    assert tokenized.stop_token_ids == (0,)


def test_tokenizer_process_can_suppress_text_while_preserving_usage() -> None:
    ingress = Queue()
    runtime_queue = Queue()
    frontend_queue = Queue()
    worker = TokenizerProcess(ingress, runtime_queue, frontend_queue, "debug")

    ingress.put(
        DetokenizeRequest(
            request_id=2,
            token_id=0,
            finished=True,
            finish_reason="stop",
            emit_text=False,
            prompt_tokens=3,
            completion_tokens=1,
        )
    )
    ingress.put(None)
    worker.run()

    chunk = frontend_queue.get(timeout=1.0)
    assert chunk.token_id == 0
    assert chunk.text == ""
    assert chunk.prompt_tokens == 3
    assert chunk.completion_tokens == 1
