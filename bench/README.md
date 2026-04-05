# bench — End-to-End Benchmark Suite

Benchmark radixInfer, vLLM, and SGLang, and generate charts that can be embedded directly into the project README.

## Files

| File | Description |
|---|---|
| `bench_e2e.py` | Core benchmark driver with no third-party runtime dependencies beyond the Python standard library |
| `plot_results.py` | Reads JSON benchmark results and generates PNG charts; requires `matplotlib` |
| `run_bench.sh` | Runs the full benchmark suite and generates plots |

## Benchmark Scenarios

### Scenario 1 — Concurrency Sweep (Three-Engine Comparison)

Use fixed input/output token counts and sweep concurrency across `[1, 4, 8, 16, 32]` to compare:

- Output throughput (tokens/s)
- TTFT (time to first token)
- TPOT (time per output token)
- End-to-end latency p50/p90/p99

### Scenario 2 — Burst / Max-Concurrency Run (Three-Engine Comparison)

Use `200` total requests and set concurrency to `200` as well, so all requests are pushed into the engine at once. This is intended to measure peak throughput under an uncapped burst load and compare radixInfer, vLLM, and SGLang at the highest request fan-in for the run.

The default burst setting is:

- `NUM_REQUESTS = 200`
- `BURST_CONC = 200`

You can override either value through environment variables when running `run_bench.sh`.

### Scenario 3 — Input-Length Sweep (radixInfer)

Use fixed concurrency `= 16` and output length `= 256` tokens, then sweep input length across `[128, 512, 1024, 2048]` to observe throughput changes under longer contexts.

### Scenario 4 — Prefix-Cache Benefit (radixInfer)

Make all requests share a common 400-token prefix and compare throughput and TTFT with and without prefix-cache hits.

## Quick Start

### 1. Install Dependencies

```bash
pip install matplotlib
```

### 2. Start the Engines

Launch the three engines in separate terminals. The example below uses `Qwen3-8B` with `2x RTX 4090 D`, `TP=2` for all engines:

```bash
# radixInfer (port 1919) — cuda:0 + cuda:1 assigned automatically
conda activate sglang
PYTHONPATH=python python -m radixinfer \
    --model Qwen/Qwen3-8B --tp-size 2 --device cuda:0 --port 1919

# vLLM (port 8000)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B --tensor-parallel-size 2 \
    --port 8000 --dtype bfloat16

# SGLang (port 30000)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --tp-size 2 \
    --port 30000 --dtype bfloat16
```

Wait until all services are ready before starting benchmarks.

### 3. Run Benchmarks

The suite benchmarks one engine at a time so multiple services do not compete for GPU memory during the same run.

**Manual start/stop mode (default):** the script pauses before and after each engine and waits for you to start or stop the service manually.

```bash
# Run engines one by one as prompted
bash bench/run_bench.sh

# Benchmark a single engine only
ENGINES=radixinfer bash bench/run_bench.sh
ENGINES=vllm       bash bench/run_bench.sh
ENGINES=sglang     bash bench/run_bench.sh

# Generate plots only after all results have been collected
bash bench/run_bench.sh --plot-only
```

**Auto start/stop mode:** set the `*_CMD` variables so the script launches and stops each service automatically.

```bash
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
              --model Qwen/Qwen3-8B --tensor-parallel-size 2 --port 8000 --dtype bfloat16" \
SGLANG_CMD="python -m sglang.launch_server \
              --model-path Qwen/Qwen3-8B --tp-size 2 --port 30000 --dtype bfloat16" \
RADIXINFER_CMD="PYTHONPATH=python python -m radixinfer \
              --model Qwen/Qwen3-8B --tp-size 2 --device cuda:0 --port 1919" \
bash bench/run_bench.sh
```

Results are written to `bench/results/`, and plots are written to `bench/results/plots/`.

These benchmark outputs are runtime artifacts and should generally not be committed.
If a chart is promoted into project documentation, copy the curated image into a stable docs asset directory such as `docs/assets/bench/`.

Burst-run outputs are written with names like:

- `bench/results/radixinfer_burst200.json`
- `bench/results/vllm_burst200.json`
- `bench/results/sglang_burst200.json`

### 4. Plot Separately

```bash
# Snapshot comparison for all three engines at concurrency = 16
python3 bench/plot_results.py \
    bench/results/vllm_c16.json \
    bench/results/sglang_c16.json \
    bench/results/radixinfer_c16.json \
    --out-dir bench/results/plots

# Concurrency sweep line chart
python3 bench/plot_results.py \
    --sweep bench/results/radixinfer_c*.json \
    --sweep-key concurrency \
    --out-dir bench/results/plots/radixinfer_sweep

# Burst comparison
python3 bench/plot_results.py \
    bench/results/vllm_burst200.json \
    bench/results/sglang_burst200.json \
    bench/results/radixinfer_burst200.json \
    --out-dir bench/results/plots/comparison_burst200

# Prefix-cache comparison
python3 bench/plot_results.py \
    --prefix-cache \
    bench/results/radixinfer_c16_nocache.json \
    bench/results/radixinfer_c16_cache.json \
    --out-dir bench/results/plots/prefix_cache
```

## Benchmark a Single Engine Manually

```bash
# Benchmark an already running service
python3 bench/bench_e2e.py \
    --engine generic \
    --base-url http://127.0.0.1:1919/v1 \
    --model Qwen/Qwen3-0.6B \
    --concurrency 16 \
    --num-requests 200 \
    --input-tokens 512 \
    --output-tokens 256 \
    --result-json bench/results/radixinfer_c16.json

# Launch a service automatically and benchmark it
python3 bench/bench_e2e.py \
    --engine vllm \
    --server-command "python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --port 8000" \
    --base-url http://127.0.0.1:8000/v1 \
    --ready-timeout 300 \
    --concurrency 16 \
    --num-requests 200 \
    --result-json bench/results/vllm_c16.json
```

## Key Arguments

| Argument | Description |
|---|---|
| `--engine` | `vllm` / `sglang` / `generic` (`generic` is used for radixInfer) |
| `--base-url` | OpenAI-compatible API base URL of the target engine |
| `--concurrency` | Number of concurrent requests |
| `--num-requests` | Total request count |
| `--input-tokens` | Input length; supports ranges such as `512:1024` |
| `--output-tokens` | Output length; supports ranges such as `128:256` |
| `--warmup-requests` | Warmup request count; excluded from final metrics |
| `--result-json` | Output path for JSON summary results |
| `--result-jsonl` | Output path for per-request detail records |
| `--dataset` | Use a JSONL dataset instead of synthetic prompts |

## Test Environment

| Item | Value |
|---|---|
| GPU | 2× NVIDIA GeForce RTX 4090 D (24 GB) |
| TP | 2 (tensor parallel) |
| Model | Qwen/Qwen3-8B |
| Precision | bfloat16 |
| Input length | 512 tokens |
| Output length | 256 tokens |
| Concurrency sweep | 1 / 4 / 8 / 16 / 32 |
| Requests | 200 per run |
| Warmup | 8 requests |
