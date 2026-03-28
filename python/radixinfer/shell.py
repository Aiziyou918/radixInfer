from __future__ import annotations

import argparse

from radixinfer.config import ServerConfig
from radixinfer.engine import build_engine
from radixinfer.engine.base import DecodeInput
from radixinfer.env import ENV
from radixinfer.transport.tokenizer_worker import create_tokenizer_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="radixInfer shell")
    parser.add_argument("--model", default="debug")
    parser.add_argument("--engine", choices=["dummy", "hf"], default="hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=ENV.SHELL_MAX_TOKENS.value)
    args = parser.parse_args()

    tokenizer = create_tokenizer_backend(None if args.model == "debug" else args.model)
    engine = build_engine(ServerConfig(model=args.model, engine_kind=args.engine, device=args.device))
    print("radixInfer shell. type /exit to quit.")
    while True:
        prompt = input("> ")
        if prompt.strip() == "/exit":
            return
        token_ids = tokenizer.encode(prompt)
        generated_ids: list[int] = []
        for _ in range(args.max_tokens):
            token_id = engine.decode(DecodeInput(request_ids=[0], token_ids=[token_ids])).next_token_ids[0]
            token_ids.append(token_id)
            generated_ids.append(token_id)
            stop_ids = list(getattr(tokenizer, "stop_token_ids", ()))
            if token_id in stop_ids:
                break
        texts = tokenizer.batch_decode([generated_ids])
        print(texts[0])


if __name__ == "__main__":
    main()
