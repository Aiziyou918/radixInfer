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


@dataclass(frozen=True)
class TokenizeRequest:
    request_id: int
    prompt: str
    sampling: SamplingParams


@dataclass(frozen=True)
class TokenizedRequest:
    request_id: int
    token_ids: list[int]
    sampling: SamplingParams


@dataclass(frozen=True)
class AbortRequest:
    request_id: int


@dataclass(frozen=True)
class DetokenizeRequest:
    request_id: int
    token_id: int
    finished: bool = False
    finish_reason: Literal["stop", "abort", "length", "error", "running"] = "running"


@dataclass(frozen=True)
class StreamChunk:
    request_id: int
    text: str
    finished: bool
    finish_reason: Literal["stop", "abort", "length", "error", "running"] = "running"


@dataclass(frozen=True)
class EngineStep:
    request_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finished: list[bool] = field(default_factory=list)
