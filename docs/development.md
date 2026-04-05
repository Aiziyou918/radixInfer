# radixInfer Development Guide

## Environment

Python 3.10+ required. The conda environment is `radixInfer`.

```bash
conda activate radixInfer
export PYTHONPATH=python
```

Core dependencies (declared in `pyproject.toml`):

- `fastapi`, `uvicorn`, `pydantic>=2.0`
- `torch`, `transformers>=4.40`
- `pyzmq`

Development dependency: `pytest>=8.0`

---

## Local Run Commands

### HTTP server (debug model, no GPU needed)

```bash
python -m radixinfer --model debug --device cpu --port 1919
```

Useful for validating: route wiring, request lifecycle, streaming behavior, frontend/backend process orchestration.

### Shell mode

```bash
python -m radixinfer --model debug --device cpu --shell
```

Shell built-ins: `/reset` (clear history), `/exit` (quit).

### Real model (GPU required)

```bash
python -m radixinfer --model Qwen/Qwen3-0.6B --device cuda:0
python -m radixinfer --model meta-llama/Llama-2-7b --device cuda:0 --tp-size 2
```

### Common CLI flags

| Flag | Description |
|---|---|
| `--model` | Model path or identifier (`debug` for testing) |
| `--device` | `cpu`, `auto`, or `cuda:N` |
| `--port` | HTTP port (default `1919`) |
| `--tp-size` | Tensor-parallel size |
| `--dist-port` | Distributed rendezvous port (auto-selected if unavailable) |
| `--page-size` | KV page size in tokens |
| `--num-pages` | Total KV pages (overrides memory estimate) |
| `--max-prefill-length` | Prefill token budget per scheduling tick |
| `--disable-zmq` | Use `mp.Queue` instead of ZMQ |

---

## Test Commands

```bash
pytest -q                              # fast default suite
pytest tests/test_prefix_cache.py -v  # single module, verbose
pytest tests/ -k "evict"              # filter by name
```

Default suite excludes:
- `tests/test_api_server.py`
- `tests/test_multiprocess_server.py`

Run the full suite explicitly when validating integration paths:

```bash
pytest tests/ -q
```

### Compile check

```bash
python -m compileall python/ -q
```

---

## Test Coverage

| File | Covers |
|---|---|
| `test_page_pool.py` | Page reservation, sharing, release |
| `test_prefix_cache.py` | Radix tree match, insert, eviction, ref-counting, heap correctness |
| `test_tokenizer_backend.py` | Tokenizer backend metadata, EOS propagation |
| `test_api_*.py` | FastAPI endpoint behavior with fake app state |
| `test_multiprocess_server.py` | Full multi-process request path (slow, excluded by default) |

---

## Documentation Map

| File | Purpose |
|---|---|
| `README.md` | Project overview, quick start |
| `docs/architecture.md` | Subsystem design, algorithms, data flow |
| `docs/developer-guide.md` | Internals, module descriptions, reading order |
| `docs/api-guide.md` | HTTP endpoint reference with request/response shapes |
| `docs/development.md` | This file — local workflow and test commands |
| `CLAUDE.md` | Claude Code context: commands, module map, key design notes |
