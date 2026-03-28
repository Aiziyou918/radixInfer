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


def create_tokenizer_backend(model_name: str | None) -> Any:
    if model_name and model_name != "debug":
        try:
            return TransformersTokenizerAdapter(model_name)
        except Exception:
            pass
    return SimpleTokenizer()


@dataclass
class TokenizerProcess:
    ingress: mp.Queue
    runtime_queue: mp.Queue
    frontend_queue: mp.Queue
    model_name: str | None = None

    def run(self) -> None:
        tokenizer = create_tokenizer_backend(self.model_name)
        while True:
            try:
                message = self.ingress.get(timeout=0.1)
            except Empty:
                continue
            if message is None:
                return
            if isinstance(message, TokenizeRequest):
                self.runtime_queue.put(
                    TokenizedRequest(
                        request_id=message.request_id,
                        token_ids=tokenizer.encode(message.prompt),
                        sampling=message.sampling,
                    )
                )
            elif isinstance(message, DetokenizeRequest):
                self.frontend_queue.put(
                    StreamChunk(
                        request_id=message.request_id,
                        text=tokenizer.decode_token(message.token_id),
                        finished=message.finished,
                        finish_reason=message.finish_reason,
                    )
                )


def start_tokenizer_process(
    ingress: mp.Queue,
    runtime_queue: mp.Queue,
    frontend_queue: mp.Queue,
    model_name: str | None = None,
) -> mp.Process:
    process = mp.Process(
        target=TokenizerProcess(ingress, runtime_queue, frontend_queue, model_name).run,
        name="radixinfer-tokenizer",
        daemon=True,
    )
    process.start()
    return process
