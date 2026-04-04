"""Tokenizer worker process: tokenization + incremental detokenization.

Handles TokenizeRequest → TokenizedRequest (sent to scheduler)
and DetokenizeRequest → StreamChunk (sent to API frontend).

Uses DetokenizeManager for stateful incremental detokenization when a full
HuggingFace tokenizer is available, so UTF-8 boundaries and CJK characters
are handled correctly.  Falls back to single-token decode for debug mode.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from queue import Empty
from typing import Any

from .protocol import (
    BatchDetokenizeRequest,
    BatchStreamChunk,
    DetokenizeRequest,
    StreamChunk,
    TokenizeRequest,
    TokenizedRequest,
)
from .tokenizer_backend import (
    SimpleTokenizer,
    TransformersTokenizerAdapter,
    create_tokenizer_backend,
)
from .queues import make_zmq_pull, make_zmq_push

# Unique sentinel used to detect "no message received yet" (distinct from None which is shutdown)
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Helper queue accessors
# ---------------------------------------------------------------------------

def _queue_get_nowait(q: Any):
    if hasattr(q, "get_nowait"):
        return q.get_nowait()
    raise Empty


def _queue_empty(q: Any) -> bool:
    if hasattr(q, "empty"):
        return q.empty()
    return False


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

@dataclass
class TokenizerProcess:
    """Tokenizer worker that handles both tokenization and incremental detokenization.

    ingress: receives TokenizeRequest (from API) and DetokenizeRequest (from runtime)
    runtime_queue: sends TokenizedRequest to the scheduler
    frontend_queue: sends StreamChunk to the API frontend
    """

    ingress: Any  # mp.Queue or ZMQ pull queue
    runtime_queue: Any  # mp.Queue or ZMQ push queue
    frontend_queue: Any  # mp.Queue or ZMQ push queue
    model_name: str | None = None

    def run(self) -> None:
        from radixinfer.transport.detokenize import DetokenizeManager

        tokenizer = create_tokenizer_backend(self.model_name)
        detok_manager = DetokenizeManager(tokenizer)

        while True:
            # --- Blocking wait for at least one message ---
            message = _SENTINEL  # sentinel to detect "nothing yet"
            while message is _SENTINEL:
                try:
                    if hasattr(self.ingress, "get_nowait"):
                        message = self.ingress.get_nowait()
                    else:
                        message = self.ingress.get(timeout=0.1)
                except Empty:
                    time.sleep(0.001)
                    continue
                except Exception as exc:
                    raise RuntimeError(
                        f"Tokenizer worker failed while reading ingress queue: {type(exc).__name__}: {exc}"
                    ) from exc

            # --- Collect all pending messages (non-blocking drain) ---
            pending: list = [message]
            while not _queue_empty(self.ingress):
                try:
                    pending.append(_queue_get_nowait(self.ingress))
                except Empty:
                    break

            # --- Separate by type; detect shutdown sentinel ---
            tokenize_msgs: list[TokenizeRequest] = []
            detokenize_msgs: list[DetokenizeRequest] = []
            shutdown = False

            for msg in pending:
                if msg is None:
                    shutdown = True
                    break  # process already-collected messages, then exit
                if isinstance(msg, TokenizeRequest):
                    tokenize_msgs.append(msg)
                elif isinstance(msg, BatchDetokenizeRequest):
                    detokenize_msgs.extend(msg.requests)
                elif isinstance(msg, DetokenizeRequest):
                    detokenize_msgs.append(msg)

            # --- Tokenize (immediately, one by one) ---
            for msg in tokenize_msgs:
                if msg.messages is not None:
                    token_ids = tokenizer.encode_messages(msg.messages)
                else:
                    token_ids = tokenizer.encode(msg.prompt)
                self.runtime_queue.put(
                    TokenizedRequest(
                        request_id=msg.request_id,
                        token_ids=token_ids,
                        sampling=msg.sampling,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        stop_token_ids=getattr(tokenizer, "stop_token_ids", ()),
                    )
                )

            # --- Incremental detokenize (batch) ---
            if detokenize_msgs:
                texts = detok_manager.detokenize(detokenize_msgs)
                chunks = [
                    StreamChunk(
                        request_id=req.request_id,
                        token_id=req.token_id,
                        text=text,
                        finished=req.finished,
                        finish_reason=req.finish_reason,
                        prompt_tokens=req.prompt_tokens,
                        completion_tokens=req.completion_tokens,
                    )
                    for req, text in zip(detokenize_msgs, texts)
                ]
                if len(chunks) == 1:
                    self.frontend_queue.put(chunks[0])
                else:
                    self.frontend_queue.put(BatchStreamChunk(chunks=chunks))

            if shutdown:
                return


def start_tokenizer_process(
    ingress: Any,
    runtime_queue: Any,
    frontend_queue: Any,
    model_name: str | None = None,
) -> mp.Process:
    """Spawn a tokenizer worker process.

    Accepts mp.Queue objects or ZMQ queue objects (or address strings).
    """
    process = mp.Process(
        target=_run_tokenizer_process,
        args=(ingress, runtime_queue, frontend_queue, model_name),
        name="radixinfer-tokenizer",
        daemon=True,
    )
    process.start()
    return process


def _run_tokenizer_process(
    ingress: Any,
    runtime_queue: Any,
    frontend_queue: Any,
    model_name: str | None,
) -> None:
    if isinstance(ingress, str):
        ingress = make_zmq_pull(ingress, create=True)
    if isinstance(runtime_queue, str):
        runtime_queue = make_zmq_push(runtime_queue, create=False)
    if isinstance(frontend_queue, str):
        frontend_queue = make_zmq_push(frontend_queue, create=False)
    TokenizerProcess(ingress, runtime_queue, frontend_queue, model_name).run()
