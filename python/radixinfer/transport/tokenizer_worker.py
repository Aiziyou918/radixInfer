from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Any

from .protocol import DetokenizeRequest, StreamChunk, TokenizeRequest, TokenizedRequest


class SimpleTokenizer:
    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = 1
        self.eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        result = []
        for ch in text:
            if ch not in self._char_to_id:
                token_id = self._next_id
                self._next_id += 1
                self._char_to_id[ch] = token_id
                self._id_to_char[token_id] = ch
            result.append(self._char_to_id[ch])
        return result or [0]

    def decode_token(self, token_id: int) -> str:
        if token_id == 0:
            return ""
        return self._id_to_char.get(token_id, "?")

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return (self.eos_token_id,)


class TransformersTokenizerAdapter:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._cache: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens or [self.tokenizer.eos_token_id or 0]

    def decode_token(self, token_id: int) -> str:
        if token_id not in self._cache:
            self._cache[token_id] = self.tokenizer.decode([token_id], skip_special_tokens=False)
        return self._cache[token_id]

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        candidates = []
        for token_id in (
            self.tokenizer.eos_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ):
            if token_id is not None:
                candidates.append(int(token_id))
        return tuple(dict.fromkeys(candidates))


def create_tokenizer_backend(model_name: str | None) -> Any:
    if model_name and model_name != "debug":
        try:
            return TransformersTokenizerAdapter(model_name)
        except Exception:
            pass
    return SimpleTokenizer()


def _queue_get_nowait(q: Any):
    """Unified get_nowait that works for mp.Queue, ZMQ queues, and queue.Queue."""
    if hasattr(q, "get_nowait"):
        return q.get_nowait()
    raise Empty


def _queue_empty(q: Any) -> bool:
    if hasattr(q, "empty"):
        return q.empty()
    return False


@dataclass
class TokenizerProcess:
    """Tokenizer worker that accepts either mp.Queue or ZMQ queue objects.

    ingress: receives TokenizeRequest (from API) and DetokenizeRequest (from runtime)
    runtime_queue: receives TokenizedRequest destined for the scheduler
    frontend_queue: receives StreamChunk destined for the API frontend
    """

    ingress: Any  # mp.Queue or ZMQ pull queue
    runtime_queue: Any  # mp.Queue or ZMQ push queue
    frontend_queue: Any  # mp.Queue or ZMQ push queue
    model_name: str | None = None

    def run(self) -> None:
        tokenizer = create_tokenizer_backend(self.model_name)
        while True:
            try:
                message = self.ingress.get(timeout=0.1) if hasattr(self.ingress, "get") else None
            except Empty:
                continue
            except Exception:
                continue
            if message is None:
                return
            if isinstance(message, TokenizeRequest):
                self.runtime_queue.put(
                    TokenizedRequest(
                        request_id=message.request_id,
                        token_ids=tokenizer.encode(message.prompt),
                        sampling=message.sampling,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        stop_token_ids=getattr(tokenizer, "stop_token_ids", ()),
                    )
                )
            elif isinstance(message, DetokenizeRequest):
                text = tokenizer.decode_token(message.token_id) if message.emit_text else ""
                self.frontend_queue.put(
                    StreamChunk(
                        request_id=message.request_id,
                        token_id=message.token_id,
                        text=text,
                        finished=message.finished,
                        finish_reason=message.finish_reason,
                        prompt_tokens=message.prompt_tokens,
                        completion_tokens=message.completion_tokens,
                    )
                )


def start_tokenizer_process(
    ingress: Any,
    runtime_queue: Any,
    frontend_queue: Any,
    model_name: str | None = None,
) -> mp.Process:
    """Spawn a daemon tokenizer process.

    Accepts mp.Queue objects or ZMQ queue objects for all three queues.
    """
    process = mp.Process(
        target=TokenizerProcess(ingress, runtime_queue, frontend_queue, model_name).run,
        name="radixinfer-tokenizer",
        daemon=True,
    )
    process.start()
    return process
