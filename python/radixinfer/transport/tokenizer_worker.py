from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Iterable

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


@dataclass
class TokenizerProcess:
    ingress: mp.Queue
    runtime_queue: mp.Queue
    frontend_queue: mp.Queue

    def run(self) -> None:
        tokenizer = SimpleTokenizer()
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
                        finish_reason="stop" if message.finished else "running",
                    )
                )


def start_tokenizer_process(
    ingress: mp.Queue,
    runtime_queue: mp.Queue,
    frontend_queue: mp.Queue,
) -> mp.Process:
    process = mp.Process(
        target=TokenizerProcess(ingress, runtime_queue, frontend_queue).run,
        name="radixinfer-tokenizer",
        daemon=True,
    )
    process.start()
    return process
