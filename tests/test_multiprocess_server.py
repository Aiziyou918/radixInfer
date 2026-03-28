from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx


def _wait_for_server_ready(proc: subprocess.Popen[str], timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    logs: list[str] = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            logs.append(line.rstrip())
            if "Uvicorn running on" in line:
                return
        elif proc.poll() is not None:
            break
    raise RuntimeError(f"server not ready, rc={proc.poll()}, logs={logs}")


def test_real_multiprocess_server_path() -> None:
    port = int(os.environ.get("RADIXINFER_TEST_PORT", "19301"))
    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    cmd = [
        sys.executable,
        "-m",
        "radixinfer.serve",
        "--model",
        "debug",
        "--engine",
        "dummy",
        "--device",
        "cpu",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd="/home/hyp/radixInfer/radixInfer",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        _wait_for_server_ready(proc)
        with httpx.Client(trust_env=False, timeout=20.0) as client:
            models = client.get(f"http://127.0.0.1:{port}/v1/models")
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "debug"

            nonstream = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "debug", "prompt": "hi", "max_tokens": 3, "stream": False},
            )
            assert nonstream.status_code == 200
            body = nonstream.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["finish_reason"] in {"stop", "length"}
            assert body["usage"]["completion_tokens"] == 3

            with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "debug",
                    "prompt": "hi",
                    "max_tokens": 3,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as stream:
                assert stream.status_code == 200
                lines = [line for line in stream.iter_lines() if line]
            joined = "\n".join(lines)
            assert "chat.completion.chunk" in joined
            assert '"usage"' in joined
            assert "[DONE]" in joined
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
