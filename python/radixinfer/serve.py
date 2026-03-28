from __future__ import annotations

import argparse

import uvicorn

from .api.server import create_app
from .config import ServerConfig


def parse_args() -> ServerConfig:
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
    parser.add_argument("--start-method", choices=["spawn", "inline"], default="spawn")
    parser.add_argument("--disable-zmq", dest="use_zmq", action="store_false")
    parser.set_defaults(use_zmq=True)
    args = parser.parse_args()
    return ServerConfig(**vars(args))


def main() -> None:
    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
