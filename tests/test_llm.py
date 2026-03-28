from radixinfer.llm import LLM
from radixinfer.transport.protocol import SamplingParams


def test_llm_generate_debug_returns_batch_outputs() -> None:
    llm = LLM(model="debug", engine="hf", device="cpu")
    outputs = llm.generate(["hello", "world"], SamplingParams(max_tokens=4))
    assert len(outputs) == 2
    assert all(isinstance(text, str) for text in outputs)
