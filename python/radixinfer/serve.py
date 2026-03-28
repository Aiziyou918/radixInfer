from __future__ import annotations

import argparse
import asyncio

import uvicorn

from .api.server import AppState, create_app
from .config import ServerConfig


def parse_args() -> tuple[ServerConfig, bool]:
    parser = argparse.ArgumentParser(description="radixInfer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--model", default="debug")
    parser.add_argument("--tokenizer-workers", type=int, default=1)
    parser.add_argument("--runtime-workers", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-prefill-length", dest="max_prefill_tokens", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", dest="total_pages", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--engine", dest="engine_kind", choices=["dummy", "hf", "real"], default="hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-zmq", dest="use_zmq", action="store_false")
    parser.add_argument("--shell", action="store_true", help="Run interactive shell instead of HTTP server")
    parser.set_defaults(use_zmq=True)
    args = parser.parse_args()
    run_shell: bool = args.shell
    del args.shell
    return ServerConfig(**vars(args)), run_shell


async def _run_shell(config: ServerConfig) -> None:
    from radixinfer.env import ENV
    from radixinfer.transport.protocol import SamplingParams, StreamChunk

    state = AppState(config=config)
    state.start()
    await state.start_listener()

    print("radixInfer shell. /exit to quit, /reset to clear history.")
    history: list[tuple[str, str]] = []

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input == "/exit":
                break
            if user_input == "/reset":
                history = []
                print("History cleared.")
                continue

            messages = [
                *[msg for user_msg, asst_msg in history
                  for msg in (
                      {"role": "user", "content": user_msg},
                      {"role": "assistant", "content": asst_msg},
                  )],
                {"role": "user", "content": user_input},
            ]

            request_id = state.next_request_id()
            output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
            state.listeners[request_id] = output_queue
            state.submit_request(
                request_id,
                user_input,
                SamplingParams(
                    max_tokens=ENV.SHELL_MAX_TOKENS.value,
                    temperature=ENV.SHELL_TEMPERATURE.value,
                    top_k=ENV.SHELL_TOP_K.value,
                    top_p=ENV.SHELL_TOP_P.value,
                ),
                messages=messages,
            )

            cur_text = ""
            try:
                while True:
                    chunk = await output_queue.get()
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        cur_text += chunk.text
                    if chunk.finished:
                        break
                print()
                history.append((user_input, cur_text))
            finally:
                state.listeners.pop(request_id, None)
    finally:
        await state.shutdown()


def main() -> None:
    import os

    config, run_shell = parse_args()
    if run_shell:
        asyncio.run(_run_shell(config))
    else:
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    # Force-exit to terminate daemon subprocesses and ZMQ I/O threads immediately.
    # Without this, Python's atexit/thread-join machinery can hang indefinitely.
    os._exit(0)


if __name__ == "__main__":
    main()
