from __future__ import annotations

import argparse
import asyncio
import socket

import uvicorn

from .api.server import create_app
from .config import ServerConfig
from .server import FrontendManager


def _is_tcp_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_runtime_dist_port(server_port: int, requested_port: int | None) -> int:
    if requested_port is not None and _is_tcp_port_available(requested_port):
        return requested_port

    preferred_port = server_port + 1
    if _is_tcp_port_available(preferred_port):
        return preferred_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_args() -> tuple[ServerConfig, bool]:
    parser = argparse.ArgumentParser(description="radixInfer server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--model", default="debug")
    parser.add_argument("--tokenizer-workers", type=int, default=1)
    parser.add_argument("--runtime-workers", type=int, default=1)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-prefill-length", dest="max_prefill_tokens", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--num-pages", dest="total_pages", type=int, default=None,
                        help="Total KV cache pages (default: auto based on available GPU memory)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tp-size", dest="tp_size", type=int, default=1)
    parser.add_argument(
        "--dist-port",
        dest="dist_port",
        type=int,
        default=None,
        help="Master port for torch.distributed rendezvous; defaults to server port + 1 or another free local port.",
    )
    parser.add_argument("--disable-zmq", dest="use_zmq", action="store_false")
    parser.add_argument("--shell", action="store_true", help="Run interactive shell instead of HTTP server")
    parser.set_defaults(use_zmq=True)
    args = parser.parse_args()
    run_shell: bool = args.shell
    del args.shell
    args.dist_port = _pick_runtime_dist_port(args.port, args.dist_port)
    return ServerConfig(**vars(args)), run_shell


async def _run_shell(config: ServerConfig) -> None:
    from radixinfer.env import ENV
    from radixinfer.transport.protocol import SamplingParams, StreamChunk

    state = FrontendManager(config=config)
    try:
        state.start_backend()
        await state.start_listener()

        print("radixInfer shell. /exit to quit, /reset to clear history.")
        history: list[tuple[str, str]] = []

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

    except KeyboardInterrupt:
        print()
    finally:
        await state.shutdown()


def main() -> None:
    import os

    config, run_shell = parse_args()
    try:
        if run_shell:
            asyncio.run(_run_shell(config))
        else:
            state = FrontendManager(config=config)
            state.start_backend()
            app = create_app(config, state=state, manage_backend=False)
            uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
