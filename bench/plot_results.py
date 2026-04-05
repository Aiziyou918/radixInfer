"""
Plot benchmark results from bench_e2e.py JSON output.

Usage
-----
# Compare multiple engines at a single concurrency level:
python plot_results.py results/vllm_c16.json results/sglang_c16.json results/radixinfer_c16.json

# Concurrency scaling sweep for one or more engines:
python plot_results.py --sweep results/radixinfer_c*.json --sweep-key concurrency

# Prefix-cache comparison (two radixInfer runs):
python plot_results.py --prefix-cache results/radixinfer_nocache.json results/radixinfer_cache.json

All plots are saved to --out-dir (default: results/plots/).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional matplotlib import
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_result(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    # bench_e2e.py wraps everything under "summary"
    if "summary" in data:
        return data["summary"]
    return data


def engine_label(path: str) -> str:
    """Derive a short label from the filename, e.g. 'vllm_c16' → 'vllm c=16'."""
    stem = Path(path).stem
    stem = re.sub(r"_c(\d+)$", r" c=\1", stem)
    stem = re.sub(r"_burst(\d+)$", r" burst=\1", stem)
    return stem


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
BAR_WIDTH = 0.22

def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    out_dir: str,
    filename: str,
    color: str = "#4C72B0",
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    x = range(len(labels))
    bars = ax.bar(x, values, color=color, width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, filename)


def _grouped_bar(
    group_labels: list[str],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_dir: str,
    filename: str,
) -> None:
    n_groups = len(group_labels)
    n_series = len(series)
    width = 0.7 / n_series
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.8), 4))
    x = range(n_groups)
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x], vals,
            width=width, label=name,
            color=COLORS[i % len(COLORS)], zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, filename)


def _line_chart(
    x_vals: list[float | int],
    series: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: str,
    filename: str,
    x_log: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, vals) in enumerate(series.items()):
        ax.plot(x_vals, vals, marker="o", label=name, color=COLORS[i % len(COLORS)], zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if x_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, filename)


# ---------------------------------------------------------------------------
# Plot modes
# ---------------------------------------------------------------------------

def plot_comparison(paths: list[str], out_dir: str) -> None:
    """Bar charts comparing multiple engines/runs side by side."""
    results = [(engine_label(p), load_result(p)) for p in paths]
    labels = [r[0] for r in results]

    def _get(r, *keys):
        d = r
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    metrics = [
        ("output_throughput_tps",   "Output throughput (tokens/s)",    "throughput_tps.png",  "#4C72B0"),
        ("request_throughput_rps",  "Request throughput (req/s)",       "throughput_rps.png",  "#55A868"),
    ]
    for key, ylabel, fname, color in metrics:
        vals = [_get(r, key) or 0.0 for _, r in results]
        if any(v > 0 for v in vals):
            _bar_chart(labels, vals, ylabel, ylabel, out_dir, fname, color)

    # TTFT grouped (mean / p50 / p90)
    ttft_series: dict[str, list[float]] = {"mean": [], "p50": [], "p90": []}
    for _, r in results:
        ttft = r.get("ttft_s") or {}
        ttft_series["mean"].append((ttft.get("mean") or 0) * 1000)
        ttft_series["p50"].append((ttft.get("p50") or 0) * 1000)
        ttft_series["p90"].append((ttft.get("p90") or 0) * 1000)
    if any(v > 0 for v in ttft_series["mean"]):
        _grouped_bar(labels, ttft_series, "Time to First Token (ms)", "TTFT (ms)", out_dir, "ttft.png")

    # TPOT
    tpot_vals = [((r.get("tpot_s") or {}).get("mean") or 0) * 1000 for _, r in results]
    if any(v > 0 for v in tpot_vals):
        _bar_chart(labels, tpot_vals, "Time Per Output Token (ms)", "TPOT (ms)", out_dir, "tpot.png", "#DD8452")

    # Latency p50 / p90 / p99
    lat_series: dict[str, list[float]] = {"p50": [], "p90": [], "p99": []}
    for _, r in results:
        lat = r.get("latency_s") or {}
        lat_series["p50"].append(lat.get("p50") or 0)
        lat_series["p90"].append(lat.get("p90") or 0)
        lat_series["p99"].append(lat.get("p99") or 0)
    if any(v > 0 for v in lat_series["p50"]):
        _grouped_bar(labels, lat_series, "End-to-End Latency (s)", "Latency (s)", out_dir, "latency.png")

    print(f"\nComparison charts written to {out_dir}/")


def plot_sweep(paths: list[str], sweep_key: str, out_dir: str) -> None:
    """Line charts for a concurrency or input-length sweep."""
    # Group files by engine name (prefix before the sweep key pattern)
    engine_data: dict[str, list[tuple[float, dict]]] = {}
    for p in sorted(paths):
        r = load_result(p)
        label = engine_label(p)
        # Try to extract sweep value from summary or filename
        x_val = r.get(sweep_key) or r.get("concurrency") or r.get("input_tokens")
        if x_val is None:
            # Fall back: parse from filename, e.g. _c16 or _i512
            m = re.search(r"[_\-](?:c|i|concurrency|input)(\d+)", Path(p).stem)
            x_val = int(m.group(1)) if m else 0
        # engine name = stem without trailing _cN / _iN
        eng = re.sub(r"[_\-](?:c|i|concurrency|input)\d+$", "", Path(p).stem)
        engine_data.setdefault(eng, []).append((float(x_val), r))

    for eng in engine_data:
        engine_data[eng].sort(key=lambda t: t[0])

    x_vals = sorted({t[0] for pts in engine_data.values() for t, _ in [(t, None) for t in pts]})

    def _series(key, *subkeys):
        out = {}
        for eng, pts in engine_data.items():
            vals = []
            for _, r in pts:
                v = r
                for k in (key, *subkeys):
                    v = (v or {}).get(k) if isinstance(v, dict) else None
                vals.append((v or 0))
            out[eng] = vals
        return out

    x_label = "Concurrency" if "c" in sweep_key else "Input tokens"
    _line_chart(
        [int(v) for v in x_vals],
        _series("output_throughput_tps"),
        f"Output Throughput vs {x_label}",
        x_label, "Throughput (tokens/s)", out_dir, f"sweep_throughput.png",
    )
    _line_chart(
        [int(v) for v in x_vals],
        {eng: [(r.get("ttft_s") or {}).get("mean", 0) * 1000 for _, r in pts]
         for eng, pts in engine_data.items()},
        f"TTFT (mean) vs {x_label}",
        x_label, "TTFT (ms)", out_dir, f"sweep_ttft.png",
    )
    _line_chart(
        [int(v) for v in x_vals],
        {eng: [(r.get("tpot_s") or {}).get("mean", 0) * 1000 for _, r in pts]
         for eng, pts in engine_data.items()},
        f"TPOT (mean) vs {x_label}",
        x_label, "TPOT (ms)", out_dir, f"sweep_tpot.png",
    )
    print(f"\nSweep charts written to {out_dir}/")


def plot_prefix_cache(paths: list[str], out_dir: str) -> None:
    """Specialized chart for prefix-cache hit vs no-hit comparison."""
    assert len(paths) == 2, "Pass exactly 2 files: [no-cache, cache]"
    labels = ["no prefix cache", "prefix cache hit"]
    results = [load_result(p) for p in paths]

    metrics = [
        ("output_throughput_tps", "Output throughput (tokens/s)", "cache_throughput.png", ["#DD8452", "#55A868"]),
        ("ttft_mean_ms",          "TTFT mean (ms)",               "cache_ttft.png",       ["#DD8452", "#55A868"]),
        ("tpot_mean_ms",          "TPOT mean (ms)",               "cache_tpot.png",       ["#DD8452", "#55A868"]),
    ]
    for key, ylabel, fname, colors in metrics:
        if key == "output_throughput_tps":
            vals = [r.get("output_throughput_tps") or 0 for r in results]
        elif key == "ttft_mean_ms":
            vals = [(r.get("ttft_s") or {}).get("mean", 0) * 1000 for r in results]
        else:
            vals = [(r.get("tpot_s") or {}).get("mean", 0) * 1000 for r in results]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, vals, color=colors, width=0.4, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Prefix Cache Effect — {ylabel}")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, out_dir, fname)

    print(f"\nPrefix-cache charts written to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_MPL:
        print("ERROR: matplotlib is required.  pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Plot bench_e2e.py results")
    parser.add_argument("files", nargs="*", help="Result JSON files for comparison mode")
    parser.add_argument("--sweep", nargs="+", metavar="FILE",
                        help="JSON files for concurrency/input-length sweep")
    parser.add_argument("--sweep-key", default="concurrency",
                        help="Metric swept over (default: concurrency)")
    parser.add_argument("--prefix-cache", nargs=2, metavar="FILE",
                        help="Two JSON files: [no-cache, with-cache]")
    parser.add_argument("--out-dir", default="results/plots",
                        help="Output directory for PNG files (default: results/plots)")
    args = parser.parse_args()

    if args.prefix_cache:
        plot_prefix_cache(args.prefix_cache, args.out_dir)
    elif args.sweep:
        expanded = []
        for p in args.sweep:
            expanded.extend(sorted(glob.glob(p)) or [p])
        plot_sweep(expanded, args.sweep_key, args.out_dir)
    elif args.files:
        expanded = []
        for p in args.files:
            expanded.extend(sorted(glob.glob(p)) or [p])
        plot_comparison(expanded, args.out_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
