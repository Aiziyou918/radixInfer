from __future__ import annotations

from typing import Any


class SimpleTokenizer:
    """Character-level debug tokenizer (no real model needed)."""

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = 1
        self.eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        result = []
        for ch in text:
            if ch not in self._char_to_id:
                token_id = self._next_id
                self._next_id += 1
                self._char_to_id[ch] = token_id
                self._id_to_char[token_id] = ch
            result.append(self._char_to_id[ch])
        return result or [0]

    def encode_messages(self, messages: list) -> list[int]:
        text = " ".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        return self.encode(text)

    def decode_token(self, token_id: int) -> str:
        if token_id == 0:
            return ""
        return self._id_to_char.get(token_id, "?")

    def batch_decode(self, token_id_lists: list[list[int]]) -> list[str]:
        return ["".join(self.decode_token(t) for t in ids) for ids in token_id_lists]

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return (self.eos_token_id,)


class TransformersTokenizerAdapter:
    """Wraps a HuggingFace tokenizer with the interface expected by tokenizer worker."""

    def __init__(self, model_name: str) -> None:
        from radixinfer.utils.hf import load_tokenizer

        self.tokenizer = load_tokenizer(model_name)
        self._cache: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens or [self.tokenizer.eos_token_id or 0]

    def encode_messages(self, messages: list) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return tokens or [self.tokenizer.eos_token_id or 0]

    def decode_token(self, token_id: int) -> str:
        if token_id not in self._cache:
            self._cache[token_id] = self.tokenizer.decode([token_id], skip_special_tokens=False)
        return self._cache[token_id]

    def batch_decode(self, token_id_lists: list[list[int]]) -> list[str]:
        return self.tokenizer.batch_decode(token_id_lists, skip_special_tokens=False)

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        candidates = []
        for token_id in (
            self.tokenizer.eos_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        ):
            if token_id is not None:
                candidates.append(int(token_id))
        return tuple(dict.fromkeys(candidates))


def create_tokenizer_backend(model_name: str | None) -> Any:
    if model_name and model_name != "debug":
        try:
            return TransformersTokenizerAdapter(model_name)
        except Exception:
            pass
    return SimpleTokenizer()


__all__ = [
    "SimpleTokenizer",
    "TransformersTokenizerAdapter",
    "create_tokenizer_backend",
]
