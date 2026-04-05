# radixInfer Architecture

## Purpose

`radixInfer` is an LLM serving system with explicit subsystem boundaries. Its design goal is to keep API handling, transport, runtime scheduling, cache management, and engine execution cleanly separated so that each layer can be understood and extended independently.

The architecture is centered on:

- A multi-process execution model with clear process boundaries
- A scheduler-driven runtime with overlap-scheduled prefill/decode batching
- A radix-tree prefix cache with heap-based LRU eviction
- A pluggable engine and model layer

---

## Subsystems

### API (`api/`)

Owns HTTP surface:

- FastAPI route registration and request validation (Pydantic schemas)
- Stop-sequence normalization and truncation
- SSE frame rendering for streaming responses
- JSON envelope construction for non-streaming responses

Does not own model execution, tokenizer state, or scheduling logic.

### Server (`server/`)

Owns frontend orchestration:

- Backend worker process startup and shutdown
- Per-request listener lifecycle and request-ID allocation
- Submission of tokenize requests to the tokenizer ingress
- Abort signaling for disconnected or stop-truncated streams

`FrontendManager` is the bridge between the API layer and the backend worker topology.

### Transport (`transport/`)

Defines the message contracts that cross process boundaries:

| Message | Direction | Purpose |
|---|---|---|
| `TokenizeRequest` | API → tokenizer | Prompt text + sampling params |
| `TokenizedRequest` | tokenizer → scheduler | Token IDs + EOS metadata |
| `AbortRequest` | API → scheduler | Cancel an in-flight request |
| `DetokenizeRequest` | scheduler → tokenizer | Generated token IDs to decode |
| `StreamChunk` | tokenizer → API | Decoded text fragment |

Also implements queue/ZMQ adapters, tokenizer backend selection, and incremental detokenization.

### Runtime (`runtime/`)

Owns request execution state and scheduling:

- Request admission and chunked-prefill tracking
- Decode-first batch construction
- Mixed-batch formation (decode + prefill in one forward pass)
- Cache and table resource coordination
- Overlap scheduling between GPU execution and CPU post-processing

The scheduler is the central state machine for token generation. See [Scheduler Design](#scheduler-design) below.

### Cache (`cache/`)

Provides logical page allocation and prefix reuse:

- `PagePool` — fixed-size physical page allocator
- `RadixPrefixCache` — radix-tree prefix cache with heap-based LRU eviction
- `CacheManager` — runtime-facing orchestration (match, allocate, insert, evict)

See [Cache Design](#cache-design) below.

### Engine (`engine/`)

Owns device execution:

- CUDA stream and device initialization
- Distributed process-group setup for tensor parallelism
- Model construction and weight loading
- KV cache allocation and page-table management
- Attention backend selection (FlashInfer / FlashAttention / HF fallback)
- CUDA graph capture and replay (`GraphRunner`)
- Token sampling

---

## Process Topology

```
┌─────────────────────────────────────────────────────┐
│  Main process                                       │
│  FastAPI / Uvicorn  +  FrontendManager              │
└──────────────┬──────────────────────────────────────┘
               │ ZMQ / mp.Queue
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼─────────────────────────────┐
│  Tokenizer  │  │  Runtime worker (rank 0)            │
│  process    │  │  Scheduler  +  Engine               │
└─────────────┘  └──────────────────────┬─────────────┘
                                        │  (TP > 1)
                               ┌────────▼─────────────┐
                               │  Runtime worker rank1 │
                               │  Scheduler  +  Engine │
                               └───────────────────────┘
```

In HTTP mode, all workers start as daemon subprocesses. The main process never calls the model directly. The tokenizer process never interacts with the GPU.

---

## End-to-End Request Flow

```
1. Client  →  POST /v1/chat/completions
2. API      →  FrontendManager.submit(TokenizeRequest)
3. Tokenizer worker  →  encodes prompt  →  emits TokenizedRequest
4. Scheduler  →  admits request into PrefillManager.pending_list
5. Scheduler  →  allocates table slot + prefix-cache pages
6. Engine    →  forward_batch(prefill)  →  sample first token
7. Scheduler  →  inserts prefix into RadixPrefixCache
8. Scheduler  →  moves request to DecodeManager
9. Engine    →  forward_batch(decode)  →  sample next token  [repeat]
10. Scheduler  →  emits DetokenizeRequest per token
11. Tokenizer worker  →  decodes token  →  emits StreamChunk
12. FrontendManager  →  routes chunk to per-request queue
13. API      →  returns SSE frame or collects final JSON
```

---

## Scheduler Design

### Batch selection policy: decode-first

The scheduler always drains decode-ready requests before admitting new prefill work. This keeps already-running requests generating tokens immediately and prevents head-of-line blocking from large prefills.

```python
decode_batch = decode_manager.schedule_next_batch()      # drain first
prefill_batch = prefill_manager.schedule_next_batch(budget)  # use remaining budget
```

If both are non-empty, they are merged into a `mixed` batch with decode requests listed first (CUDA-graph-friendly ordering for the eventual decode-only steady state).

### Overlap scheduling

The scheduler runs two alternating branches:

**Non-overlap mode** (simpler, higher latency):
```
receive → schedule → forward → process results → [repeat]
```

**Overlap mode** (default):
```
iteration N:   receive → schedule → launch forward(N) → process results(N-1)
iteration N+1: receive → schedule → launch forward(N+1) → process results(N)
```

GPU work for batch N runs concurrently with CPU post-processing of batch N-1. The previous `ForwardOutput` is stored as `last_data` and consumed at the start of the next iteration.

### TP-rank determinism

`DecodeManager` stores running requests in a `dict[uid, Req]`. `schedule_next_batch` always sorts by `uid` before building the `Batch`. This ensures identical token positions across TP ranks, preventing allreduce mismatches that would cause NCCL deadlock under sustained load.

### Chunked prefill

When a request's input length exceeds the prefill token budget, `PrefillAdder` creates a `ChunkedReq` that carries the partial prefix. The request stays in `pending_list` and is continued in subsequent ticks until the full prompt is prefilled.

---

## Cache Design

### Page allocation

The KV cache is divided into fixed-size logical pages (`page_size` tokens each). `CacheManager.free_slots` is a GPU tensor of page-aligned free slot indices. Allocation pops from the front; freeing appends to the back.

`lazy_free_region()` defers frees into a list and batches them into a single `torch.cat` after the loop, avoiding O(n) repeated tensor allocations inside the post-processing loop.

### Radix prefix cache

`RadixPrefixCache` is a radix tree where each node stores:

- `_key`: torch tensor of token IDs (the prefix fragment at this node)
- `_value`: torch tensor of page indices (KV slots for the corresponding tokens)
- `ref_count`: number of active requests sharing this node
- `timestamp`: last access time (nanoseconds), used for LRU ordering

**Lookup** (`match_prefix`): walks the tree comparing `key_fn(input_ids[offset:])` at each level, splitting nodes on partial matches.

**Insert** (`insert_prefix`): walks to the deepest match point, appends the unmatched suffix as a new leaf node.

**Ref-counting** (`lock_handle` / `unlock`): walks from the matched leaf up to root, incrementing or decrementing `ref_count`. A node is evictable only when `ref_count == 0 and is_leaf()`.

**Heap-based LRU eviction**: evictable leaf nodes are maintained in a `_evictable_heap` — a min-heap of `(timestamp_at_push, node)` tuples. New evictable nodes are pushed when:
- a node's `ref_count` drops to 0 (after unlock) and it is a leaf
- a new leaf is created by `insert_prefix`
- a cache-hit in `_tree_walk` refreshes a node's timestamp

Entries are validated lazily on pop: a popped entry is skipped if its stored timestamp no longer matches the node's current timestamp (meaning the node was recently accessed and a fresh entry was pushed), or if the node is no longer evictable.

This makes eviction O(k log h) instead of O(n) full-tree DFS, where k is the number of pages to evict and h is the heap size.

### CacheManager hot-path optimization

`CacheManager` pre-allocates two pinned host buffers (`_pt_row_buf`, `_pt_pos_buf`) in `__init__` at size `num_pages * page_size`. The `_write_page_table` method reuses these buffers as scatter indices for H2D page-table writes, avoiding `cudaHostAlloc` on every batch.

---

## Model Architecture

### Shared decoder factory

All Llama-family models (Llama, Mistral, Qwen2, Qwen3) share the same decoder structure: `RopeAttn` + `GatedMLP` per layer, `VocabParallelEmbedding`, `RMSNorm`, `ParallelLMHead`. They differ only in attention kwargs.

`models/_decoder.py` provides `build_decoder_model(attn_kwargs)` which returns a `ForCausalLM` class with `DecoderLayer`, `DecoderModel`, and `ForCausalLM` as closures, forwarding `attn_kwargs` to `RopeAttn`:

| Model | `attn_kwargs` |
|---|---|
| Llama, Mistral | `{}` |
| Qwen2 | `{"has_qk_norm": False, "has_attn_bias": True}` |
| Qwen3 | `{"has_qk_norm": True}` |

Qwen3-MoE uses a separate implementation in `models/qwen3_moe.py`.

### Model registration

`models/register.py` maps HF architecture names to `(module, class)` pairs. `create_model(model_config)` calls `get_model_class(architectures[0], model_config)` which lazy-imports the module and instantiates the class.

---

## Transport Modes

| Mode | When | Notes |
|---|---|---|
| ZMQ | Default | Separate OS sockets; works across processes with different Python objects |
| `mp.Queue` | `--disable-zmq` | In-process; simpler but requires shared memory |

Both modes preserve the same logical protocol (same message types, same ordering guarantees).

---

## Current Maturity

| Area | Status |
|---|---|
| API / transport / scheduling / cache | Substantially complete |
| Radix prefix cache | Complete, heap-based LRU |
| CUDA graph | Complete (`engine/graph.py`) |
| FlashAttention / FlashInfer backends | HF fallback only; real backends are stubs |
| Tensor parallelism plumbing | Complete; edge cases under ZMQ overlap TBD |
| OpenAI API compatibility | Partial; `n > 1` not supported |
