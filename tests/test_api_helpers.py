from radixinfer.api.server import _build_completion_payload, _build_stream_payload


def test_build_stream_payload_shape() -> None:
    payload = _build_stream_payload(3, {"content": "x"}, None)
    assert payload["id"] == "chatcmpl-3"
    assert payload["choices"][0]["delta"]["content"] == "x"


def test_build_completion_payload_shape() -> None:
    payload = _build_completion_payload(5, "debug", "hello", "stop")
    assert payload["model"] == "debug"
    assert payload["choices"][0]["message"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] == "stop"
