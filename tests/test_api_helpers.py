from radixinfer.api.server import _build_completion_payload, _build_stream_payload
from radixinfer.server.common import build_usage


def test_build_stream_payload_shape() -> None:
    payload = _build_stream_payload(3, "debug", 123, {"content": "x"}, None, build_usage(2, 1))
    assert payload["id"] == "chatcmpl-3"
    assert payload["model"] == "debug"
    assert payload["created"] == 123
    assert payload["choices"][0]["delta"]["content"] == "x"
    assert payload["usage"]["total_tokens"] == 3


def test_build_completion_payload_shape() -> None:
    payload = _build_completion_payload(5, "debug", 456, "hello", "stop", build_usage(4, 2))
    assert payload["model"] == "debug"
    assert payload["created"] == 456
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["prompt_tokens"] == 4
