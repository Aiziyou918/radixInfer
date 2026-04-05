# radixInfer

[English](README.md) | [简体中文](README.zh-CN.md)

`radixInfer` is a layered LLM serving system with a runnable end-to-end control plane. It keeps API handling, transport, runtime scheduling, cache management, and engine execution clearly separated.

## Features

- FastAPI-based HTTP service with SSE streaming
- OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints
- Interactive shell mode for local debugging
- Multi-process architecture: API, tokenizer, and runtime in separate processes
- Decode-first overlap scheduler (prefill/decode/mixed batches)
- Radix-tree prefix cache with heap-based LRU eviction
- Paged KV cache with shared-prefix support
- Llama-family model factory (Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE)
- Pluggable attention backends (FlashInfer / FlashAttention / HF fallback)
- Tensor parallelism support

## Repository Structure

Main packages under `python/radixinfer`:

| Package | Purpose |
|---|---|
| `api` | HTTP routes, schemas, SSE/JSON rendering |
| `server` | Frontend orchestration, listener lifecycle, backend startup |
| `transport` | ZMQ/queue adapters, protocol types, tokenizer worker, detokenization |
| `runtime` | Scheduler, prefill/decode managers, cache manager, runtime I/O |
| `cache` | Page pool, radix prefix cache, KV pool |
| `engine` | Model execution, CUDA graph, sampling, distributed setup |
| `models` | Model registry, configs, weights, model implementations |
| `distributed` | Tensor-parallel communication helpers |
| `layers` | Transformer building blocks (attention, linear, norm, embedding) |

## Quick Start

### Requirements

Python 3.10+. Core dependencies: `fastapi`, `pyzmq`, `torch`, `transformers>=4.40`, `uvicorn`, `pydantic>=2.0`.

### Run the Server

```bash
# Debug model (no GPU, no model download)
PYTHONPATH=python python -m radixinfer --model debug --device cpu --port 1919

# Real model on GPU
PYTHONPATH=python python -m radixinfer --model Qwen/Qwen3-0.6B --device cuda:0
```

### Shell Mode

```bash
PYTHONPATH=python python -m radixinfer --model debug --device cpu --shell
```

Shell commands: `/reset` (clear history), `/exit` (quit).

### Run Tests

```bash
pytest -q
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | HTTP bind host |
| `--port` | `1919` | HTTP port |
| `--model` | `debug` | Model name or path |
| `--device` | `auto` | `cpu`, `auto`, or `cuda:N` |
| `--tp-size` | `1` | Tensor-parallel size |
| `--num-pages` | derived | Total KV cache pages |
| `--page-size` | `16` | Tokens per KV page |
| `--max-prefill-length` | `2048` | Prefill token budget per tick |
| `--dist-port` | auto | Distributed rendezvous port |
| `--disable-zmq` | off | Use `mp.Queue` instead of ZMQ |
| `--shell` | off | Launch shell mode |

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /v1/models` | Model listing |
| `POST /generate` | Simple generation (plain SSE or JSON) |
| `POST /v1/completions` | OpenAI text completion |
| `POST /v1/chat/completions` | OpenAI chat completion |

All generation endpoints support streaming (`stream=true`) and non-streaming. `n > 1` is not currently supported.

## Architecture Overview

```
HTTP Request
  → API Server (FastAPI + SSE)
  → FrontendManager → TokenizerWorker [separate process]
  → [ZMQ or mp.Queue]
  → Scheduler [separate process]
      DecodeManager  — dict[uid, Req], decode-first drain
      PrefillManager — prefix cache match, chunked prefill
      CacheManager   — paged alloc, lazy-free, H2D scatter
      RadixPrefixCache — radix tree + heap-based LRU eviction
  → Engine
      Model (Llama / Qwen2 / Qwen3 / Qwen3-MoE / Mistral)
      KV Cache (paged MHA)
      CUDA Graph Runner
      Sampler
  → DetokenizeRequest → TokenizerWorker → StreamChunk → SSE
```

The scheduler runs an overlap loop by default: GPU execution for batch N overlaps with CPU post-processing of batch N-1.

## Documentation

- [简体中文 README](README.zh-CN.md)
- [Architecture](docs/architecture.md) — subsystem design, algorithms, data flow
- [Developer Guide](docs/developer-guide.md) — internals, module descriptions, reading order
- [API Guide](docs/api-guide.md) — endpoint reference with request/response shapes
- [Development Guide](docs/development.md) — local workflow, test commands

## Current Status

The control plane (API, transport, scheduling, cache) is substantially complete and end-to-end runnable. Remaining work:

- Real FlashAttention/FlashInfer attention backends (HF fallback currently)
- ZMQ overlap scheduler edge cases under sustained TP load
- Full OpenAI API compatibility (`n > 1`, broader schema coverage)
