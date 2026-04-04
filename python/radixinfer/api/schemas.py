from __future__ import annotations

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    class BaseModel:  # type: ignore[no-redef]
        pass


class ChatMessage(BaseModel):
    role: str
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False
    ignore_eos: bool = False
    stream_options: StreamOptions | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    stream: bool = True
    ignore_eos: bool = False
    stop: str | list[str] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = True
    ignore_eos: bool = False
    stream_options: StreamOptions | None = None


__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "GenerateRequest",
    "StreamOptions",
]
