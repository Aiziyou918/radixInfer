from __future__ import annotations

import argparse
import time

from radixinfer.engine.base import DecodeInput
from radixinfer.engine.dummy import DummyEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="radixInfer synthetic benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args()

    engine = DummyEngine()
    tokens = [list(range(args.prompt_len)) for _ in range(args.batch_size)]
    request_ids = list(range(args.batch_size))

    start = time.time()
    for _ in range(args.steps):
        output = engine.decode(DecodeInput(request_ids=request_ids, token_ids=tokens))
        for seq, token_id in zip(tokens, output.next_token_ids, strict=True):
            seq.append(token_id)
    elapsed = time.time() - start
    total = args.batch_size * args.steps
    print(f"tokens={total} elapsed={elapsed:.4f}s throughput={total / max(elapsed, 1e-9):.2f} tok/s")


if __name__ == "__main__":
    main()
