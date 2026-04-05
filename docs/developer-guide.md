# radixInfer Developer Guide

## Overview

`radixInfer` is a layered LLM serving codebase with a runnable end-to-end control plane. It keeps API handling, transport, runtime scheduling, cache management, and engine execution in clearly separated subsystems.

At a high level, the project provides:

- FastAPI HTTP endpoints (streaming SSE + non-streaming JSON)
- An interactive shell mode for local debugging
- A tokenizer/detokenizer worker process
- A scheduler with overlap-scheduled prefill, decode, and mixed batches
- A radix-tree prefix cache with heap-based LRU eviction
- A paged KV cache and pluggable attention backends
- A model factory for Llama-family architectures

---

## Repository Layout

The Python package lives under `python/radixinfer`.

| Package | Responsibility |
|---|---|
| `api` | HTTP schemas, SSE/JSON rendering, FastAPI routes |
| `server` | Frontend orchestration, listener management, backend startup/shutdown |
| `transport` | Cross-process protocol objects, ZMQ/queue adapters, tokenizer backend, detokenization |
| `runtime` | Scheduler, prefill/decode managers, cache-manager integration, runtime I/O |
| `cache` | Logical page allocation, radix prefix cache, page-pool abstractions |
| `engine` | CUDA/GPU execution, graph runner, attention backends, sampling, distributed setup |
| `models` | Model registry, config, weight loading, model implementations |
| `distributed` | Tensor-parallel communication helpers |
| `layers` | Shared transformer building blocks (attention, linear, norm, embedding) |
| `tests` | Unit-style and integration-style tests |

---

## System Architecture

See `docs/architecture.md` for the full subsystem and lifecycle description.

The short version: requests flow from HTTP → FrontendManager → TokenizerWorker → Scheduler → Engine → TokenizerWorker → HTTP. Each boundary is a message-passing interface; no layer reaches directly into another layer's internals.

---

## Process Model

Three process types run in HTTP mode:

| Process | Runs | Owns |
|---|---|---|
| Main | FastAPI / Uvicorn | FrontendManager, per-request listener queues |
| Tokenizer | Separate process | HF tokenizer state, incremental detokenization |
| Runtime (× TP) | Separate process(es) | Scheduler, Engine, KV cache |

Design principle: API-facing code never calls the model; runtime code never formats HTTP responses; tokenizer state stays centralized.

---

## Entrypoints and Startup

```bash
python -m radixinfer          # resolves to radixinfer.serve.main()
```

Main startup modes:

- **HTTP server**: starts backend workers, builds FastAPI app, runs Uvicorn
- **Shell mode** (`--shell`): starts backend workers, runs interactive loop

Important CLI arguments:

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | HTTP bind host |
| `--port` | `1919` | HTTP port |
| `--model` | `debug` | Model path or identifier |
| `--device` | `auto` | Device selector (`auto`, `cpu`, `cuda:N`) |
| `--tp-size` | `1` | Tensor-parallel world size |
| `--max-running-requests` | `256` | Maximum in-flight requests |
| `--max-prefill-length` | `2048` | Prefill token budget per tick |
| `--page-size` | `16` | Tokens per KV page |
| `--num-pages` | derived | Total KV pages (overrides memory estimate) |
| `--dist-port` | auto | Distributed rendezvous port |
| `--disable-zmq` | off | Use `mp.Queue` instead of ZMQ |
| `--shell` | off | Launch shell mode |

---

## Request Lifecycle

### 1. API ingress

`radixinfer.api.server.create_app()` registers the routes.

Current endpoints: `GET /v1/models`, `POST /generate`, `POST /v1/completions`, `POST /v1/chat/completions`.

### 2. Normalization

The API layer converts HTTP payloads to internal `SamplingParams` and a normalized stop-sequence list. For chat requests, the tokenizer receives structured `messages`; for completions, a plain prompt string.

### 3. Frontend submission

`FrontendManager`:
- allocates a `request_id`
- opens a per-request output `asyncio.Queue`
- sends a `TokenizeRequest` to the tokenizer ingress

### 4. Tokenization

The tokenizer process encodes the prompt or chat messages, emits a `TokenizedRequest` with `token_ids` and EOS/stop metadata, and forwards it to the scheduler.

For `--model debug`, a built-in tokenizer is used so no HF model download is needed.

### 5. Scheduling and execution

The scheduler:

1. Receives the `TokenizedRequest` and adds it to `PrefillManager.pending_list`
2. Matches the prefix against `RadixPrefixCache`; copies cached pages into the table slot
3. Allocates new KV pages for the uncached suffix
4. On each tick, builds a decode-first batch (decode requests drain first, then prefill)
5. Calls `engine.forward_batch()` and samples next tokens
6. Post-processes: updates prefix cache, frees finished requests, emits `DetokenizeRequest`

### 6. Detokenization and streaming

The tokenizer worker converts `DetokenizeRequest` messages to `StreamChunk` objects. The frontend listener routes them to the correct per-request queue, and the API layer streams SSE frames or assembles the final JSON.

---

## Runtime Internals

### Scheduler

`runtime/scheduler.py` — `Scheduler(SchedulerIOMixin)`.

Two execution paths:
- `overlap_loop()` — default; overlaps GPU forward with CPU post-processing of previous batch
- `normal_loop()` — sequential; simpler but higher latency

Batch policy: decode-first. `schedule_next_batch()` calls `decode_manager.schedule_next_batch()` first, then `prefill_manager.schedule_next_batch(budget)`, and merges if both are non-empty.

Hot-path buffers: four pre-allocated pinned tensors (`_pos_buf`, `_input_row_buf`, `_write_row_buf`, `_write_pos_buf`) avoid `cudaHostAlloc` on every batch. Size is bounded by `prefill_budget + max_running_req`.

### DecodeManager

`runtime/decode.py` — `DecodeManager`.

Stores running requests as `dict[uid, Req]` (not a set):
- `abort_req(uid)` is O(1) via `dict.pop`
- `filter_reqs` avoids creating a new set on every tick
- `schedule_next_batch` sorts by `uid` for TP-rank determinism

### PrefillManager + PrefillAdder

`runtime/prefill.py`.

`PrefillManager` holds `pending_list: list[PendingReq]`. On each tick, `schedule_next_batch(budget)` creates a `PrefillAdder` and iterates the pending list:

1. For each `PendingReq`, `PrefillAdder.try_add_one()` checks:
   - table slot available
   - prefix cache match
   - estimated memory fits within budget
2. If the prompt is too long for the current budget, a `ChunkedReq` is created and the request is continued in the next tick
3. The batch list is rebuilt: chunked requests go to the front of `pending_list`

### CacheManager

`runtime/cache_manager.py`.

Key design choices:

- `free_slots` is a GPU int32 tensor of page-aligned free slot indices
- `allocate_paged` computes needed pages per request, calls `_allocate`, then `_write_page_table`
- `_write_page_table` uses pre-allocated pinned host buffers (`_pt_row_buf`, `_pt_pos_buf`) to build H2D scatter indices without per-batch memory allocation
- `lazy_free_region()` defers `_free` calls into `_deferred_frees` and batches them into a single `torch.cat` after the loop
- `cache_req` handles both in-progress (lock new handle) and finished (free remaining pages) cases

### TableManager

`runtime/table.py`.

Simple slot allocator: a Python list `_free_slots = [0, 1, ..., max_running_req-1]`. `allocate()` pops from the back (O(1)); `free(slot)` appends (O(1) amortized). `token_pool` mirrors `page_table` shape on GPU so input IDs can be gathered in one indexed operation.

---

## Cache Internals

### RadixPrefixCache

`cache/prefix_store.py`.

**Tree structure**: each `RadixTreeNode` holds:
- `_key`: token IDs for this prefix segment (torch.int32 tensor, CPU)
- `_value`: KV page indices (torch.int32 tensor, CPU)
- `ref_count`: number of active requests sharing this node
- `timestamp`: nanosecond access time for LRU ordering
- `children`: `dict[key_fn(node._key), RadixTreeNode]`

`key_fn` maps a tensor to a hashable key: `x[0].item()` for `page_size=1`, `tuple(x[:page_size].tolist())` otherwise.

**Tree walk** (`_tree_walk`):
1. At each node, look up `children[key_fn(input_ids[prefix_len:])]`
2. If found, call `node.get_match_len(...)` — uses `fast_compare_key` kernel
3. If partial match, call `node.split_at(match_len)` — inserts a new parent node
4. On full match, update `node.timestamp` and push a fresh heap entry if the node is an evictable leaf

**Eviction heap** (`_evictable_heap`):
- Min-heap of `(timestamp_at_push, node)` tuples
- Nodes are pushed when they become evictable (ref_count→0 and is_leaf), when new leaves are created, and when cache-hit timestamps are refreshed
- On pop, the entry is validated: skip if `ts != node.timestamp` (stale), `ref_count != 0` (re-locked), or `not is_leaf()` (gained children)
- After evicting a node, its parent is pushed if it becomes an evictable leaf
- Complexity: O(k log h) per eviction call (k = pages to evict, h = heap size)

**Ref-counting** (`lock_handle` / `unlock`):
- `lock`: walk leaf→root, `ref_count += 1`; transition 0→1 moves tokens from evictable to protected
- `unlock`: walk leaf→root, `ref_count -= 1`; transition 1→0 moves tokens to evictable, pushes to heap if leaf

---

## Model Layer

### Decoder factory

`models/_decoder.py` — `build_decoder_model(attn_kwargs)`.

Returns a `ForCausalLM` class built from three inner classes:

```
ForCausalLM
  └── DecoderModel
        ├── VocabParallelEmbedding
        ├── OPList[DecoderLayer × num_layers]
        │     ├── RopeAttn(**attn_kwargs)
        │     ├── GatedMLP
        │     ├── RMSNormFused (input)
        │     └── RMSNormFused (post-attn)
        └── RMSNormFused (final)
  └── ParallelLMHead
```

Each model file is 3 lines:

```python
from ._decoder import build_decoder_model
LlamaForCausalLM = build_decoder_model()
__all__ = ["LlamaForCausalLM"]
```

### Registration

`models/register.py` maps HF architecture name strings to `(module_path, class_name)` pairs. `get_model_class` lazy-imports the module and instantiates the class. To add a new model: add a mapping to `_MODEL_REGISTRY` and create a model file.

---

## API Behavior

### `/generate`

Minimal generation endpoint. Returns plain text SSE frames or a JSON with `{text, finish_reason, usage}`.

### `/v1/completions`

OpenAI text-completion envelope. Returns `text_completion` / `text_completion.chunk`. `n > 1` is rejected.

### `/v1/chat/completions`

OpenAI chat-completion envelope. Accepts `messages` (structured) or `prompt` (fallback). Returns `chat.completion` / `chat.completion.chunk`. `n > 1` is rejected.

### Stop sequences

Normalized in the API layer: `null` → no stop, `str` → `(str,)`, `list` → filtered tuple. Applied at the API layer before returning; the backend sees stop token IDs, not strings.

---

## Development Workflow

### Run the server

```bash
conda activate sglang
export PYTHONPATH=python
python -m radixinfer --model debug --device cpu --port 1919
```

### Run shell mode

```bash
python -m radixinfer --model debug --device cpu --shell
```

Shell commands: `/reset` (clear history), `/exit` (quit).

### Run tests

```bash
pytest -q                              # fast suite (default)
pytest tests/test_prefix_cache.py -v  # specific module
```

By default, `tests/test_api_server.py` and `tests/test_multiprocess_server.py` are excluded from the fast suite.

### Compile check

```bash
python -m compileall python/ -q
```

---

## Testing Coverage

| Test file | What it covers |
|---|---|
| `test_page_pool.py` | Page allocation, reservation sharing, release |
| `test_prefix_cache.py` | Radix tree match, insert, eviction, ref-counting |
| `test_tokenizer_backend.py` | Tokenizer backend metadata and EOS handling |
| `test_api_*.py` | FastAPI route behavior with fake app state |
| `test_multiprocess_server.py` | Full multi-process request path (excluded from fast suite) |

---

## Known Limitations

- OpenAI compatibility is partial: `n > 1` unsupported, limited schema coverage
- Attention backends: HF fallback only; FlashInfer/FlashAttention stubs exist but are not complete
- Heavier integration tests excluded from the default `pytest -q` run
- Real model execution is GPU-centric; `--device cpu` only works with `--model debug`

---

## Reading Order for New Contributors

1. `README.md`
2. `docs/architecture.md`
3. `python/radixinfer/core.py` — core data structures (`Req`, `Batch`, `SamplingParams`)
4. `python/radixinfer/serve.py` — entrypoint
5. `python/radixinfer/api/server.py` — HTTP surface
6. `python/radixinfer/server/frontend.py` — request lifecycle from API side
7. `python/radixinfer/transport/protocol.py` — message contracts
8. `python/radixinfer/runtime/scheduler.py` — scheduling loop
9. `python/radixinfer/cache/prefix_store.py` — prefix cache
10. `python/radixinfer/engine/engine.py` — GPU execution
