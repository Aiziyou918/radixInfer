#!/usr/bin/env bash
# run_bench.sh — run all benchmark scenarios and generate plots
#
# Usage:
#   ./bench/run_bench.sh                          # all engines, default model
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./bench/run_bench.sh
#   ENGINES=radixinfer ./bench/run_bench.sh       # single engine
#   CONCURRENCY_LIST="1 4 16" ./bench/run_bench.sh
#
# Requirements:
#   - Each engine's server must already be running, OR set *_CMD variables below.
#   - matplotlib installed for plot_results.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ENGINES="${ENGINES:-vllm sglang radixinfer}"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:30000/v1}"
RADIXINFER_URL="${RADIXINFER_URL:-http://127.0.0.1:1919/v1}"

# Server launch commands (leave empty = assume server already running)
VLLM_CMD="${VLLM_CMD:-}"
SGLANG_CMD="${SGLANG_CMD:-}"
RADIXINFER_CMD="${RADIXINFER_CMD:-}"

READY_TIMEOUT="${READY_TIMEOUT:-300}"

# Benchmark parameters — tuned for Qwen3-8B on 2× RTX 4090 D
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 4 8 16 32}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
WARMUP="${WARMUP:-8}"

RESULTS_DIR="bench/results"
PLOTS_DIR="bench/results/plots"
BENCH="python3 bench/bench_e2e.py"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper: run one scenario
# ---------------------------------------------------------------------------

run_one() {
    local engine="$1"
    local base_url="$2"
    local extra_args="${3:-}"
    local tag="${4:-}"
    local conc="$5"

    local label="${engine}_c${conc}${tag}"
    local out="$RESULTS_DIR/${label}.json"

    if [[ -f "$out" ]]; then
        echo "  [skip] $label (already exists)"
        return
    fi

    echo "  running: $label  (concurrency=$conc)"
    $BENCH \
        --engine "$engine" \
        --base-url "$base_url" \
        --model "$MODEL" \
        --concurrency "$conc" \
        --num-requests "$NUM_REQUESTS" \
        --warmup-requests "$WARMUP" \
        --input-tokens "$INPUT_TOKENS" \
        --output-tokens "$OUTPUT_TOKENS" \
        --result-json "$out" \
        $extra_args \
        || echo "  WARNING: $label failed, continuing"
}

# ---------------------------------------------------------------------------
# Scenario 1: Concurrency sweep — all engines
# ---------------------------------------------------------------------------

echo "=== Scenario 1: Concurrency sweep ==="

for engine in $ENGINES; do
    case "$engine" in
        vllm)        url="$VLLM_URL";       srv_cmd="$VLLM_CMD";       eng_arg="vllm"    ;;
        sglang)      url="$SGLANG_URL";     srv_cmd="$SGLANG_CMD";     eng_arg="sglang"  ;;
        radixinfer)  url="$RADIXINFER_URL"; srv_cmd="$RADIXINFER_CMD"; eng_arg="generic" ;;
        *)           echo "Unknown engine: $engine"; continue ;;
    esac

    # Auto-start server if command provided
    if [[ -n "$srv_cmd" ]]; then
        echo "  starting $engine..."
        mkdir -p bench/logs
        eval "$srv_cmd" > "bench/logs/${engine}.log" 2>&1 &
        SRV_PID=$!
        trap "kill $SRV_PID 2>/dev/null || true" EXIT
        $BENCH --engine "$eng_arg" --base-url "$url" --ready-timeout "$READY_TIMEOUT" \
               --num-requests 0 2>/dev/null || true  # wait-only mode
    fi

    echo "--- $engine ---"
    for conc in $CONCURRENCY_LIST; do
        run_one "$engine" "$url" "" "" "$conc"
    done

    # Kill auto-started server
    if [[ -n "${SRV_PID:-}" ]]; then
        kill "$SRV_PID" 2>/dev/null || true
        unset SRV_PID
    fi
done

# ---------------------------------------------------------------------------
# Scenario 2: Input-length sweep — radixInfer only
# ---------------------------------------------------------------------------

echo ""
echo "=== Scenario 2: Input-length sweep (radixInfer) ==="

INPUT_LEN_LIST="${INPUT_LEN_LIST:-128 512 1024 2048}"
FIXED_CONC=16

for ilen in $INPUT_LEN_LIST; do
    label="radixinfer_i${ilen}"
    out="$RESULTS_DIR/${label}.json"
    if [[ -f "$out" ]]; then echo "  [skip] $label"; continue; fi
    echo "  input_tokens=$ilen concurrency=$FIXED_CONC"
    $BENCH \
        --engine generic \
        --base-url "$RADIXINFER_URL" \
        --model "$MODEL" \
        --concurrency "$FIXED_CONC" \
        --num-requests "$NUM_REQUESTS" \
        --warmup-requests "$WARMUP" \
        --input-tokens "$ilen" \
        --output-tokens "$OUTPUT_TOKENS" \
        --result-json "$out" \
        || echo "  WARNING: $label failed"
done

# ---------------------------------------------------------------------------
# Scenario 3: Prefix-cache effect — radixInfer
# ---------------------------------------------------------------------------

echo ""
echo "=== Scenario 3: Prefix-cache hit vs no-hit (radixInfer) ==="

# No-cache: all unique prompts (default synthetic mode)
run_one "radixinfer" "$RADIXINFER_URL" "" "_nocache" 16

# Cache-hit: all requests share a long common prefix
# Use --extra-body to fix the prompt so every request reuses the same prefix
CACHE_OUT="$RESULTS_DIR/radixinfer_c16_cache.json"
if [[ ! -f "$CACHE_OUT" ]]; then
    echo "  running: prefix cache hit scenario"
    # Generate a shared prefix file inline (512 fixed tokens = one unique prompt repeated)
    CACHE_DATASET="$RESULTS_DIR/_cache_dataset.jsonl"
    python3 -c "
import json, sys
prefix = ' '.join(f'tok{i}' for i in range(400))
for i in range(${NUM_REQUESTS}):
    print(json.dumps({'prompt': prefix + f' query_{i}', 'output_tokens': ${OUTPUT_TOKENS}}))
" > "$CACHE_DATASET"
    $BENCH \
        --engine generic \
        --base-url "$RADIXINFER_URL" \
        --model "$MODEL" \
        --concurrency 16 \
        --num-requests "$NUM_REQUESTS" \
        --warmup-requests "$WARMUP" \
        --dataset "$CACHE_DATASET" \
        --result-json "$CACHE_OUT" \
        || echo "  WARNING: prefix-cache scenario failed"
fi

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

echo ""
echo "=== Generating plots ==="

# 1. Concurrency sweep comparison (all engines, mid-concurrency snapshot)
SNAP_CONC=16
SNAP_FILES=""
for engine in $ENGINES; do
    f="$RESULTS_DIR/${engine}_c${SNAP_CONC}.json"
    [[ -f "$f" ]] && SNAP_FILES="$SNAP_FILES $f"
done
if [[ -n "$SNAP_FILES" ]]; then
    python3 bench/plot_results.py $SNAP_FILES --out-dir "$PLOTS_DIR"
fi

# 2. Concurrency sweep line charts
for engine in $ENGINES; do
    files=$(ls "$RESULTS_DIR/${engine}_c"[0-9]*.json 2>/dev/null | grep -v nocache | grep -v cache || true)
    if [[ -n "$files" ]]; then
        python3 bench/plot_results.py --sweep $files --sweep-key concurrency \
            --out-dir "$PLOTS_DIR/${engine}_sweep"
    fi
done

# 3. Input-length sweep
INPUT_FILES=$(ls "$RESULTS_DIR/radixinfer_i"*.json 2>/dev/null || true)
if [[ -n "$INPUT_FILES" ]]; then
    python3 bench/plot_results.py --sweep $INPUT_FILES --sweep-key input_tokens \
        --out-dir "$PLOTS_DIR/input_sweep"
fi

# 4. Prefix-cache comparison
NO_CACHE="$RESULTS_DIR/radixinfer_c16_nocache.json"
WITH_CACHE="$RESULTS_DIR/radixinfer_c16_cache.json"
if [[ -f "$NO_CACHE" && -f "$WITH_CACHE" ]]; then
    python3 bench/plot_results.py --prefix-cache "$NO_CACHE" "$WITH_CACHE" \
        --out-dir "$PLOTS_DIR/prefix_cache"
fi

echo ""
echo "Done. Results: $RESULTS_DIR/  Plots: $PLOTS_DIR/"
