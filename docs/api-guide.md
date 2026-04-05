# radixInfer API Guide

## Overview

`radixInfer` currently exposes a small set of HTTP endpoints aimed at validating the service architecture and providing basic OpenAI-style compatibility.

Current endpoints:

- `GET /v1/models`
- `GET|POST|HEAD|OPTIONS /v1`
- `POST /generate`
- `POST /v1/completions`
- `POST /v1/chat/completions`

## General Behavior

The API layer is implemented in `radixinfer.api.server`.

Shared behavior across generation endpoints:

- request bodies are validated with Pydantic models
- streaming uses Server-Sent Events
- non-streaming collects the full response before returning JSON
- stop sequences are normalized and applied at the API layer
- request aborts are propagated if the client disconnects or a stop sequence is matched

## `GET /v1/models`

Returns a minimal list with the configured model identifier.

Example response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "debug",
      "object": "model",
      "created": 1710000000,
      "owned_by": "radixinfer",
      "root": "debug"
    }
  ]
}
```

This endpoint is a compatibility convenience, not a full registry service.

## `GET|POST|HEAD|OPTIONS /v1`

Returns a simple health-style payload:

```json
{
  "status": "ok"
}
```

## `POST /generate`

This is the simplest generation endpoint.

Request fields:

- `prompt: str`
- `max_tokens: int = 64`
- `temperature: float = 0.0`
- `top_k: int = -1`
- `top_p: float = 1.0`
- `stream: bool = true`
- `ignore_eos: bool = false`
- `stop: str | list[str] | null = null`

Example request:

```json
{
  "prompt": "Hello",
  "max_tokens": 16,
  "stream": true
}
```

Streaming behavior:

- each frame contains plain text in `data: ...`
- the stream ends with `data: [DONE]`

Non-streaming response shape:

```json
{
  "text": "generated text",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 5,
    "total_tokens": 8
  }
}
```

## `POST /v1/completions`

This endpoint returns OpenAI-style text completion envelopes.

Request fields:

- `model: str`
- `prompt: str`
- `max_tokens: int = 64`
- `temperature: float = 0.0`
- `top_k: int = -1`
- `top_p: float = 1.0`
- `n: int = 1`
- `stop: str | list[str] | null = null`
- `presence_penalty: float = 0.0`
- `frequency_penalty: float = 0.0`
- `stream: bool = false`
- `ignore_eos: bool = false`
- `stream_options: {"include_usage": bool} | null`

Current limitation:

- `n != 1` is rejected with HTTP 400

Non-streaming response shape:

```json
{
  "id": "cmpl-5",
  "object": "text_completion",
  "created": 1710000000,
  "model": "debug",
  "choices": [
    {
      "index": 0,
      "text": "hello",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 2,
    "total_tokens": 6
  }
}
```

Streaming behavior:

- frames contain `text_completion.chunk` payloads
- `usage` may be included on the final chunk when `stream_options.include_usage=true`
- the stream ends with `data: [DONE]`

## `POST /v1/chat/completions`

This endpoint returns OpenAI-style chat completion envelopes.

Request fields:

- `model: str`
- `prompt: str | null`
- `messages: list[{role, content}] | null`
- `max_tokens: int = 64`
- `temperature: float = 0.0`
- `top_k: int = -1`
- `top_p: float = 1.0`
- `n: int = 1`
- `stop: str | list[str] | null = null`
- `presence_penalty: float = 0.0`
- `frequency_penalty: float = 0.0`
- `stream: bool = true`
- `ignore_eos: bool = false`
- `stream_options: {"include_usage": bool} | null`

Input behavior:

- if `messages` is provided, structured chat messages are forwarded to the tokenizer worker
- if `messages` is missing, the API falls back to `prompt`

Current limitation:

- `n != 1` is rejected with HTTP 400

Non-streaming response shape:

```json
{
  "id": "chatcmpl-3",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "debug",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "hello"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 2,
    "total_tokens": 6
  }
}
```

Streaming behavior:

- frames contain `chat.completion.chunk` payloads
- the first delta may include `role: assistant`
- later chunks carry incremental `content`
- `usage` may be included on the final chunk when requested
- the stream ends with `data: [DONE]`

## Stop Sequences

Stop-sequence handling is normalized in the API layer:

- `null` means no stop sequence
- a single string becomes a one-item tuple
- a list is filtered for non-empty entries

The API layer also truncates generated text when a configured stop sequence appears, including during streaming.

## Usage Reporting

Usage accounting is surfaced in API responses using:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

For streaming responses, usage is only attached to the final chunk when explicitly requested through `stream_options.include_usage`.

## Compatibility Notes

The current API surface should be treated as partial compatibility, not a drop-in implementation of the full OpenAI API.

Known constraints:

- only a small endpoint set is implemented
- `n > 1` is not supported
- schema and behavior are focused on the current serving skeleton
- compatibility extensions will require further implementation work
