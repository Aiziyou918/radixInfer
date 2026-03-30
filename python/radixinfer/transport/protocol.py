from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass(frozen=True)
class TokenizeRequest:
    request_id: int
    prompt: str
    sampling: SamplingParams
    # Optional list of chat messages; when set the tokenizer applies the chat
    # template instead of encoding `prompt` directly.
    messages: list | None = None


@dataclass(frozen=True)
class TokenizedRequest:
    request_id: int
    token_ids: list[int]
    sampling: SamplingParams
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class AbortRequest:
    request_id: int


@dataclass(frozen=True)
class DetokenizeRequest:
    request_id: int
    token_id: int
    finished: bool = False
    finish_reason: Literal["stop", "abort", "length", "error", "running"] = "running"
    emit_text: bool = True
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class StreamChunk:
    request_id: int
    token_id: int
    text: str
    finished: bool
    finish_reason: Literal["stop", "abort", "length", "error", "running"] = "running"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class BatchDetokenizeRequest:
    """Wraps multiple DetokenizeRequests for efficient batch transport over ZMQ."""
    requests: list


@dataclass(frozen=True)
class BatchStreamChunk:
    """Wraps multiple StreamChunks for efficient batch transport over ZMQ."""
    chunks: list[StreamChunk]


@dataclass(frozen=True)
class EngineStep:
    request_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)
