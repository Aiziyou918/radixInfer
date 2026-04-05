#!/usr/bin/env python3
"""Generic end-to-end benchmark for OpenAI-compatible inference engines.

Supports engines such as vLLM and SGLang by targeting their OpenAI-compatible
HTTP APIs. The script can optionally launch the server process, wait for
readiness, run concurrent streaming requests, and summarize latency/throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now() -> float:
    return time.perf_counter()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def get_json(url: str, headers: Dict[str, str], timeout: float) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def stream_sse_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Iterable[Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_part = line[5:].strip()
            if data_part == "[DONE]":
                break
            try:
                yield json.loads(data_part)
            except json.JSONDecodeError:
                continue


def make_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def discover_model(base_url: str, headers: Dict[str, str], timeout: float) -> str:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "models")
    response = get_json(url, headers, timeout)
    models = response.get("data") or []
    if not models:
        raise RuntimeError(f"No models found from {url}")
    first = models[0]
    model_id = first.get("id")
    if not model_id:
        raise RuntimeError(f"Malformed /v1/models response: {response}")
    return model_id


def wait_until_ready(url: str, headers: Dict[str, str], timeout: float, interval: float) -> float:
    start = now()
    last_error = None
    while now() - start < timeout:
        try:
            get_json(url, headers, timeout=min(5.0, interval + 1.0))
            return now() - start
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(interval)
    raise RuntimeError(f"Server not ready within {timeout}s: {last_error}")


def random_span(spec: str, rng: random.Random) -> int:
    if ":" in spec:
        left, right = spec.split(":", 1)
        low = int(left)
        high = int(right)
        if low > high:
            raise ValueError(f"Invalid range: {spec}")
        return rng.randint(low, high)
    return int(spec)


def synthetic_prompt(token_count: int, seed_value: int) -> str:
    words = []
    for idx in range(token_count):
        words.append(f"tok{seed_value % 97}_{idx % 997}")
    return " ".join(words)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno} must be a JSON object")
            if "prompt" not in row:
                raise ValueError(f"{path}:{lineno} missing prompt")
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


@dataclass
class RequestSpec:
    index: int
    prompt: str
    target_output_tokens: int
    input_tokens_hint: Optional[int] = None


@dataclass
class RequestResult:
    index: int
    ok: bool
    status: str
    error: Optional[str]
    latency_s: float
    ttft_s: float
    decode_s: float
    output_tokens: int
    output_chars: int


class ServerProcess:
    def __init__(self, command: str, cwd: Optional[str], log_path: Path) -> None:
        self.command = command
        self.cwd = cwd
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None
        self._log_handle = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            shlex.split(self.command),
            cwd=self.cwd,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        if self._log_handle is not None:
            self._log_handle.close()


def build_request_specs(args: argparse.Namespace) -> List[RequestSpec]:
    rng = random.Random(args.seed)
    if args.dataset:
        rows = load_dataset(Path(args.dataset))
        specs: List[RequestSpec] = []
        for idx in range(args.num_requests):
            row = rows[idx % len(rows)]
            prompt = str(row["prompt"])
            output_tokens = int(row.get("output_tokens", random_span(args.output_tokens, rng)))
            input_hint = row.get("input_tokens")
            specs.append(
                RequestSpec(
                    index=idx,
                    prompt=prompt,
                    target_output_tokens=output_tokens,
                    input_tokens_hint=int(input_hint) if input_hint is not None else None,
                )
            )
        return specs

    specs = []
    for idx in range(args.num_requests):
        input_tokens = random_span(args.input_tokens, rng)
        output_tokens = random_span(args.output_tokens, rng)
        specs.append(
            RequestSpec(
                index=idx,
                prompt=synthetic_prompt(input_tokens, args.seed + idx),
                target_output_tokens=output_tokens,
                input_tokens_hint=input_tokens,
            )
        )
    return specs


def extract_stream_piece(chunk: Dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    if isinstance(delta.get("content"), str):
        return delta["content"]
    message = choice.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    text = choice.get("text")
    if isinstance(text, str):
        return text
    return ""


def extract_usage_output_tokens(chunk: Dict[str, Any]) -> Optional[int]:
    usage = chunk.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        return None
    try:
        return int(completion_tokens)
    except (TypeError, ValueError):
        return None


def run_one_request(
    spec: RequestSpec,
    *,
    url: str,
    headers: Dict[str, str],
    model: str,
    timeout: float,
    extra_body: Dict[str, Any],
    include_usage: bool,
) -> RequestResult:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.target_output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": include_usage},
    }
    payload.update(extra_body)

    start = now()
    first_token_at: Optional[float] = None
    output_chars = 0
    usage_output_tokens: Optional[int] = None
    try:
        for chunk in stream_sse_json(url, payload, headers, timeout):
            piece = extract_stream_piece(chunk)
            if piece and first_token_at is None:
                first_token_at = now()
            output_chars += len(piece)
            usage_tokens = extract_usage_output_tokens(chunk)
            if usage_tokens is not None:
                usage_output_tokens = usage_tokens

        end = now()
        if first_token_at is None:
            first_token_at = end
        output_tokens = usage_output_tokens if usage_output_tokens is not None else spec.target_output_tokens
        return RequestResult(
            index=spec.index,
            ok=True,
            status="ok",
            error=None,
            latency_s=end - start,
            ttft_s=first_token_at - start,
            decode_s=end - first_token_at,
            output_tokens=output_tokens,
            output_chars=output_chars,
        )
    except Exception as exc:  # noqa: BLE001
        end = now()
        ttft = (first_token_at - start) if first_token_at is not None else end - start
        decode = (end - first_token_at) if first_token_at is not None else 0.0
        return RequestResult(
            index=spec.index,
            ok=False,
            status="error",
            error=str(exc),
            latency_s=end - start,
            ttft_s=ttft,
            decode_s=decode,
            output_tokens=0,
            output_chars=0,
        )


def bench_requests(
    specs: List[RequestSpec],
    *,
    url: str,
    headers: Dict[str, str],
    model: str,
    timeout: float,
    concurrency: int,
    extra_body: Dict[str, Any],
    include_usage: bool,
) -> List[RequestResult]:
    results: List[Optional[RequestResult]] = [None] * len(specs)
    completed = 0
    completed_lock = threading.Lock()
    started_at = now()

    def task(spec: RequestSpec) -> RequestResult:
        return run_one_request(
            spec,
            url=url,
            headers=headers,
            model=model,
            timeout=timeout,
            extra_body=extra_body,
            include_usage=include_usage,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_map = {pool.submit(task, spec): spec.index for spec in specs}
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results[result.index] = result
            with completed_lock:
                completed += 1
                elapsed = now() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                print(
                    f"\rCompleted {completed}/{len(specs)} requests, {rate:.2f} req/s",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    print(file=sys.stderr)
    return [result for result in results if result is not None]


def build_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if args.engine in {"vllm", "sglang"}:
        extra_body["ignore_eos"] = True
        extra_body["top_k"] = 1
    if args.extra_body:
        extra_body.update(json.loads(args.extra_body))
    return extra_body


def print_report(
    *,
    results: List[RequestResult],
    total_wall_s: float,
    startup_s: float,
    model: str,
    engine: str,
) -> Dict[str, Any]:
    ok = [item for item in results if item.ok]
    failures = [item for item in results if not item.ok]
    latency = [item.latency_s for item in ok]
    ttft = [item.ttft_s for item in ok]
    tpot = [item.decode_s / max(item.output_tokens, 1) for item in ok if item.output_tokens > 0]
    total_output_tokens = sum(item.output_tokens for item in ok)

    summary = {
        "engine": engine,
        "model": model,
        "startup_s": startup_s,
        "total_wall_s": total_wall_s,
        "requests_total": len(results),
        "requests_ok": len(ok),
        "requests_failed": len(failures),
        "request_throughput_rps": (len(ok) / total_wall_s) if total_wall_s > 0 else 0.0,
        "output_throughput_tps": (total_output_tokens / total_wall_s) if total_wall_s > 0 else 0.0,
        "latency_s": summarize(latency),
        "ttft_s": summarize(ttft),
        "tpot_s": summarize(tpot),
        "failures": [asdict(item) for item in failures[:10]],
    }

    print("\n=== Benchmark Summary ===")
    print(f"engine                 : {engine}")
    print(f"model                  : {model}")
    print(f"startup_s              : {startup_s:.3f}")
    print(f"total_wall_s           : {total_wall_s:.3f}")
    print(f"requests               : {len(ok)}/{len(results)} ok")
    print(f"request_throughput_rps : {summary['request_throughput_rps']:.3f}")
    print(f"output_throughput_tps  : {summary['output_throughput_tps']:.3f}")
    print(
        "latency_s              : "
        f"mean={summary['latency_s']['mean']:.3f} "
        f"p50={summary['latency_s']['p50']:.3f} "
        f"p90={summary['latency_s']['p90']:.3f} "
        f"p99={summary['latency_s']['p99']:.3f}"
    )
    print(
        "ttft_s                 : "
        f"mean={summary['ttft_s']['mean']:.3f} "
        f"p50={summary['ttft_s']['p50']:.3f} "
        f"p90={summary['ttft_s']['p90']:.3f} "
        f"p99={summary['ttft_s']['p99']:.3f}"
    )
    print(
        "tpot_s                 : "
        f"mean={summary['tpot_s']['mean']:.5f} "
        f"p50={summary['tpot_s']['p50']:.5f} "
        f"p90={summary['tpot_s']['p90']:.5f} "
        f"p99={summary['tpot_s']['p99']:.5f}"
    )
    if failures:
        print("failures               :")
        for item in failures[:5]:
            print(f"  - #{item.index}: {item.error}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="generic", choices=["generic", "vllm", "sglang"])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--input-tokens", default="512:1024")
    parser.add_argument("--output-tokens", default="128:256")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--ready-timeout", type=float, default=600.0)
    parser.add_argument("--ready-interval", type=float, default=2.0)
    parser.add_argument("--server-command", default="")
    parser.add_argument("--server-cwd", default="")
    parser.add_argument("--server-log", default="logs/server.log")
    parser.add_argument("--result-json", default="")
    parser.add_argument("--result-jsonl", default="")
    parser.add_argument("--extra-body", default="")
    parser.add_argument("--disable-stream-usage", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = make_headers(args.api_key)
    base_url = args.base_url.rstrip("/") + "/"
    chat_url = urllib.parse.urljoin(base_url, "chat/completions")
    ready_url = urllib.parse.urljoin(base_url, "models")
    extra_body = build_extra_body(args)

    server = None
    startup_s = 0.0
    try:
        if args.server_command:
            server = ServerProcess(
                command=args.server_command,
                cwd=args.server_cwd or None,
                log_path=Path(args.server_log),
            )
            print(f"Launching server: {args.server_command}")
            server.start()
            startup_s = wait_until_ready(
                ready_url,
                headers,
                timeout=args.ready_timeout,
                interval=args.ready_interval,
            )
        else:
            wait_until_ready(
                ready_url,
                headers,
                timeout=min(args.ready_timeout, 30.0),
                interval=args.ready_interval,
            )

        model = args.model or discover_model(base_url, headers, args.timeout)
        print(f"Using model: {model}")

        specs = build_request_specs(args)
        warmup_count = min(args.warmup_requests, len(specs))
        if warmup_count > 0:
            print(f"Running warmup requests: {warmup_count}")
            _ = bench_requests(
                specs[:warmup_count],
                url=chat_url,
                headers=headers,
                model=model,
                timeout=args.timeout,
                concurrency=min(args.concurrency, warmup_count),
                extra_body=extra_body,
                include_usage=not args.disable_stream_usage,
            )

        print(f"Running benchmark: requests={len(specs)}, concurrency={args.concurrency}")
        started_at = now()
        results = bench_requests(
            specs,
            url=chat_url,
            headers=headers,
            model=model,
            timeout=args.timeout,
            concurrency=args.concurrency,
            extra_body=extra_body,
            include_usage=not args.disable_stream_usage,
        )
        total_wall_s = now() - started_at
        summary = print_report(
            results=results,
            total_wall_s=total_wall_s,
            startup_s=startup_s,
            model=model,
            engine=args.engine,
        )

        if args.result_json:
            path = Path(args.result_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"summary": summary, "results": [asdict(item) for item in results]}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.result_jsonl:
            path = Path(args.result_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for item in results:
                    handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        return 0 if summary["requests_failed"] == 0 else 2
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
