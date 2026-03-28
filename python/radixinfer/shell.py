from __future__ import annotations

from radixinfer.engine.base import DecodeInput
from radixinfer.engine.dummy import DummyEngine
from radixinfer.transport.tokenizer_worker import SimpleTokenizer


def main() -> None:
    tokenizer = SimpleTokenizer()
    engine = DummyEngine()
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
