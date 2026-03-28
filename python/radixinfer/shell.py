from __future__ import annotations

import argparse

from radixinfer.config import ServerConfig
from radixinfer.engine import build_engine
from radixinfer.engine.base import DecodeInput
from radixinfer.transport.tokenizer_worker import create_tokenizer_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="radixInfer shell")
    parser.add_argument("--model", default="debug")
    parser.add_argument("--engine", choices=["dummy", "hf"], default="hf")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    tokenizer = create_tokenizer_backend(None if args.model == "debug" else args.model)
    engine = build_engine(ServerConfig(model=args.model, engine_kind=args.engine, device=args.device))
    print("radixInfer shell. type /exit to quit.")
    while True:
        prompt = input("> ")
        if prompt.strip() == "/exit":
            return
        token_ids = tokenizer.encode(prompt)
        produced: list[str] = []
        for _ in range(32):
            token_id = engine.decode(DecodeInput(request_ids=[0], token_ids=[token_ids])).next_token_ids[0]
            token_ids.append(token_id)
            text = tokenizer.decode_token(token_id)
            produced.append(text)
        print("".join(produced))


if __name__ == "__main__":
    main()
