#!/usr/bin/env bash
# run_bench.sh — sequential per-engine benchmark
#
# Each engine is tested one at a time (start → bench all scenarios → stop → next).
# Two workflows:
#
#   A) Manual start/stop (default):
#      The script pauses before each engine, waits for user to start the server,
#      runs all benchmarks, then pauses again to let user stop it.
#
#      bash bench/run_bench.sh
#
#   B) Auto start/stop (set *_CMD):
#      The script launches and kills each server automatically.
#
#      VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
#                    --model Qwen/Qwen3-8B --tensor-parallel-size 2 --port 8000 --dtype bfloat16" \
#      SGLANG_CMD="python -m sglang.launch_server \
#                    --model-path Qwen/Qwen3-8B --tp-size 2 --port 30000 --dtype bfloat16" \
#      RADIXINFER_CMD="python -m radixinfer \
#                    --model Qwen/Qwen3-8B --tp-size 2 --device cuda:0 --port 1919" \
#      bash bench/run_bench.sh
#
#   C) Single engine:
#      ENGINES=radixinfer bash bench/run_bench.sh
#
#   D) Plot only (results already exist):
#      bash bench/run_bench.sh --plot-only

set -euo pipefail
cd "$(dirname "$0")/.."          # always run from repo root

PLOT_ONLY=false
[[ "${1:-}" == "--plot-only" ]] && PLOT_ONLY=true

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL="${MODEL:-Qwen/Qwen3-8B}"
ENGINES="${ENGINES:-vllm sglang radixinfer}"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1}"
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:30000/v1}"
RADIXINFER_URL="${RADIXINFER_URL:-http://127.0.0.1:1919/v1}"

# Auto-start commands (empty = manual mode)
VLLM_CMD="${VLLM_CMD:-}"
SGLANG_CMD="${SGLANG_CMD:-}"
RADIXINFER_CMD="${RADIXINFER_CMD:-}"

READY_TIMEOUT="${READY_TIMEOUT:-300}"
READY_INTERVAL="${READY_INTERVAL:-5}"

# Benchmark parameters — Qwen3-8B, 2× RTX 4090 D, TP=2
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 4 8 16 32}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
WARMUP="${WARMUP:-8}"
FIXED_CONC=16        # used for input-length and prefix-cache scenarios

RESULTS_DIR="bench/results"
PLOTS_DIR="bench/results/plots"
BENCH="python3 bench/bench_e2e.py"
PLOT="python3 bench/plot_results.py"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRV_PID=""

_start_server() {
    local engine="$1" cmd="$2" url="$3"
    if [[ -n "$cmd" ]]; then
        echo "  [auto] starting $engine..."
        mkdir -p bench/logs
        eval "$cmd" > "bench/logs/${engine}.log" 2>&1 &
        SRV_PID=$!
        echo "  [auto] PID=$SRV_PID, waiting for ready (timeout ${READY_TIMEOUT}s)..."
        $BENCH --engine generic --base-url "$url" \
               --ready-timeout "$READY_TIMEOUT" --ready-interval "$READY_INTERVAL" \
               --num-requests 0 2>/dev/null || true
        echo "  [auto] $engine ready."
    else
        echo ""
        echo "┌─────────────────────────────────────────────────────┐"
        echo "│  Please start $engine and press ENTER when ready.   │"
        echo "│  URL: $url"
        echo "└─────────────────────────────────────────────────────┘"
        read -r _
    fi
}

_stop_server() {
    local engine="$1" cmd="$2"
    if [[ -n "$cmd" && -n "$SRV_PID" ]]; then
        echo "  [auto] stopping $engine (PID=$SRV_PID)..."
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
        SRV_PID=""
        sleep 2   # let GPU memory drain before next engine
    else
        echo ""
        echo "┌─────────────────────────────────────────────────────┐"
        echo "│  $engine done. Stop the server, then press ENTER.   │"
        echo "└─────────────────────────────────────────────────────┘"
        read -r _
    fi
}

_run() {
    # _run <engine> <url> <concurrency> <input_tokens> <output_tokens> <tag> [extra bench args...]
    local engine="$1" url="$2" conc="$3" itok="$4" otok="$5" tag="$6"
    shift 6
    local label="${engine}_c${conc}${tag}"
    local out="$RESULTS_DIR/${label}.json"

    if [[ -f "$out" ]]; then
        echo "    [skip] $label"
        return
    fi
    echo "    bench: $label  (conc=$conc  in=$itok  out=$otok)"
    $BENCH \
        --engine "$engine" \
        --base-url "$url" \
        --model "$MODEL" \
        --concurrency "$conc" \
        --num-requests "$NUM_REQUESTS" \
        --warmup-requests "$WARMUP" \
        --input-tokens "$itok" \
        --output-tokens "$otok" \
        --result-json "$out" \
        "$@" \
        || echo "    WARNING: $label failed, continuing"
}

# ---------------------------------------------------------------------------
# Per-engine benchmark suite
# ---------------------------------------------------------------------------

bench_engine() {
    local engine="$1" url="$2"
    # bench_e2e uses "generic" for radixinfer; vllm/sglang use their own names
    local eng_arg="$engine"
    [[ "$engine" == "radixinfer" ]] && eng_arg="generic"

    echo ""
    echo "════════════════════════════════════════"
    echo " Benchmarking: $engine  ($url)"
    echo "════════════════════════════════════════"

    # --- Scenario 1: Concurrency sweep ---
    echo "  [1/3] Concurrency sweep: $CONCURRENCY_LIST"
    for conc in $CONCURRENCY_LIST; do
        _run "$engine" "$url" "$conc" "$INPUT_TOKENS" "$OUTPUT_TOKENS" "" --engine "$eng_arg"
    done

    # --- Scenario 2: Input-length sweep (radixinfer only) ---
    if [[ "$engine" == "radixinfer" ]]; then
        echo "  [2/3] Input-length sweep: 128 512 1024 2048"
        for ilen in 128 512 1024 2048; do
            local label="radixinfer_i${ilen}"
            local out="$RESULTS_DIR/${label}.json"
            if [[ -f "$out" ]]; then echo "    [skip] $label"; continue; fi
            echo "    bench: $label  (conc=$FIXED_CONC  in=$ilen  out=$OUTPUT_TOKENS)"
            $BENCH \
                --engine generic \
                --base-url "$url" \
                --model "$MODEL" \
                --concurrency "$FIXED_CONC" \
                --num-requests "$NUM_REQUESTS" \
                --warmup-requests "$WARMUP" \
                --input-tokens "$ilen" \
                --output-tokens "$OUTPUT_TOKENS" \
                --result-json "$out" \
                || echo "    WARNING: $label failed"
        done

        # --- Scenario 3: Prefix-cache hit vs no-hit ---
        echo "  [3/3] Prefix-cache effect"
        local nocache_out="$RESULTS_DIR/radixinfer_c${FIXED_CONC}_nocache.json"
        local cache_out="$RESULTS_DIR/radixinfer_c${FIXED_CONC}_cache.json"

        # no-cache: unique prompts (already done in concurrency sweep as _c16.json,
        # but we need a separate run with unique prompts to be explicit)
        if [[ ! -f "$nocache_out" ]]; then
            cp "$RESULTS_DIR/radixinfer_c${FIXED_CONC}.json" "$nocache_out" 2>/dev/null \
                || _run "radixinfer" "$url" "$FIXED_CONC" "$INPUT_TOKENS" "$OUTPUT_TOKENS" \
                        "_nocache" --engine generic
        fi

        # cache-hit: shared 400-token prefix
        if [[ ! -f "$cache_out" ]]; then
            local ds="$RESULTS_DIR/_cache_dataset.jsonl"
            python3 -c "
import json
prefix = ' '.join(f'tok{i}' for i in range(400))
for i in range(${NUM_REQUESTS}):
    print(json.dumps({'prompt': prefix + f' query_{i}', 'output_tokens': ${OUTPUT_TOKENS}}))
" > "$ds"
            echo "    bench: radixinfer_c${FIXED_CONC}_cache  (prefix-cache hit)"
            $BENCH \
                --engine generic \
                --base-url "$url" \
                --model "$MODEL" \
                --concurrency "$FIXED_CONC" \
                --num-requests "$NUM_REQUESTS" \
                --warmup-requests "$WARMUP" \
                --dataset "$ds" \
                --result-json "$cache_out" \
                || echo "    WARNING: prefix-cache scenario failed"
        fi
    else
        echo "  [2/3] Input-length sweep: skipped (radixinfer only)"
        echo "  [3/3] Prefix-cache effect: skipped (radixinfer only)"
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if $PLOT_ONLY; then
    echo "Skipping benchmarks, generating plots only..."
else
    for engine in $ENGINES; do
        case "$engine" in
            vllm)       url="$VLLM_URL";       cmd="$VLLM_CMD"       ;;
            sglang)     url="$SGLANG_URL";     cmd="$SGLANG_CMD"     ;;
            radixinfer) url="$RADIXINFER_URL"; cmd="$RADIXINFER_CMD" ;;
            *) echo "Unknown engine: $engine"; continue ;;
        esac

        _start_server "$engine" "$cmd" "$url"
        bench_engine "$engine" "$url"
        _stop_server "$engine" "$cmd"
    done
fi

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

echo ""
echo "════════════════════════════════════════"
echo " Generating plots"
echo "════════════════════════════════════════"

# 1. Snapshot comparison at fixed concurrency
for snap_conc in $FIXED_CONC; do
    snap_files=""
    for engine in $ENGINES; do
        f="$RESULTS_DIR/${engine}_c${snap_conc}.json"
        [[ -f "$f" ]] && snap_files="$snap_files $f"
    done
    if [[ -n "$snap_files" ]]; then
        echo "  comparison (concurrency=$snap_conc)..."
        $PLOT $snap_files --out-dir "$PLOTS_DIR/comparison_c${snap_conc}"
    fi
done

# 2. Concurrency sweep line charts per engine
for engine in $ENGINES; do
    files=$(ls "$RESULTS_DIR/${engine}_c"[0-9]*.json 2>/dev/null \
            | grep -v nocache | grep -v cache || true)
    if [[ -n "$files" ]]; then
        echo "  concurrency sweep: $engine..."
        $PLOT --sweep $files --sweep-key concurrency \
              --out-dir "$PLOTS_DIR/${engine}_conc_sweep"
    fi
done

# 3. Input-length sweep
input_files=$(ls "$RESULTS_DIR/radixinfer_i"*.json 2>/dev/null || true)
if [[ -n "$input_files" ]]; then
    echo "  input-length sweep..."
    $PLOT --sweep $input_files --sweep-key input_tokens \
          --out-dir "$PLOTS_DIR/radixinfer_input_sweep"
fi

# 4. Prefix-cache comparison
nc="$RESULTS_DIR/radixinfer_c${FIXED_CONC}_nocache.json"
wc_="$RESULTS_DIR/radixinfer_c${FIXED_CONC}_cache.json"
if [[ -f "$nc" && -f "$wc_" ]]; then
    echo "  prefix-cache effect..."
    $PLOT --prefix-cache "$nc" "$wc_" \
          --out-dir "$PLOTS_DIR/prefix_cache"
fi

echo ""
echo "Done."
echo "  Results : $RESULTS_DIR/"
echo "  Plots   : $PLOTS_DIR/"
