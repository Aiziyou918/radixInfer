from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from radixinfer.config import ServerConfig
from radixinfer.engine import build_engine
from radixinfer.engine.base import DecodeInput
from radixinfer.transport.protocol import SamplingParams
from radixinfer.transport.tokenizer_worker import create_tokenizer_backend


class LLM:
    def __init__(
        self,
        model: str = "debug",
        *,
        engine: str = "hf",
        device: str = "auto",
        default_sampling: SamplingParams | None = None,
    ) -> None:
        self.config = ServerConfig(model=model, engine_kind=engine, device=device)
        self.engine = build_engine(self.config)
        self.tokenizer = create_tokenizer_backend(None if model == "debug" else model)
        self.default_sampling = default_sampling or SamplingParams(max_tokens=64)

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling: SamplingParams | Iterable[SamplingParams] | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        sampling_list = self._normalize_sampling(len(prompt_list), sampling)
        token_batches = [self.tokenizer.encode(prompt) for prompt in prompt_list]
        generated_ids: list[list[int]] = [[] for _ in prompt_list]

        max_steps = max(params.max_tokens for params in sampling_list)
        for step in range(max_steps):
            active_indices = [
                i for i, params in enumerate(sampling_list) if len(generated_ids[i]) < params.max_tokens
            ]
            if not active_indices:
                break
            batch = DecodeInput(
                request_ids=active_indices,
                token_ids=[token_batches[i] + generated_ids[i] for i in active_indices],
            )
            output = self.engine.decode(batch)
            for idx, token_id in zip(active_indices, output.next_token_ids, strict=True):
                generated_ids[idx].append(token_id)
        return ["".join(self.tokenizer.decode_token(token_id) for token_id in tokens) for tokens in generated_ids]

    def _normalize_sampling(
        self,
        batch_size: int,
        sampling: SamplingParams | Iterable[SamplingParams] | None,
    ) -> list[SamplingParams]:
        if sampling is None:
            return [replace(self.default_sampling) for _ in range(batch_size)]
        if isinstance(sampling, SamplingParams):
            return [replace(sampling) for _ in range(batch_size)]
        sampling_list = list(sampling)
        if len(sampling_list) != batch_size:
            raise ValueError("Sampling list size must match prompt batch size")
        return sampling_list
