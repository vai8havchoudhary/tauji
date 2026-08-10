import httpx
import pytest

from tauji.provider import CLIProxyProvider


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "non-object"),
        ({"id": "x"}, "at least one choice"),
        ({"choices": []}, "at least one choice"),
        (
            {"choices": [{"message": {}, "finish_reason": "stop"}]},
            "neither content nor tool calls",
        ),
        (
            {"choices": [{"message": {"content": "ok"}, "finish_reason": None}]},
            "finish_reason",
        ),
    ],
)
async def test_malformed_provider_response_is_an_error(payload: object, error: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CLIProxyProvider("http://proxy/v1", "key", client=client)
    events = [
        event
        async for event in provider.stream_response(
            model="model",
            system="system",
            messages=[],
            tools=[],
        )
    ]
    assert events[-1].reason == "error"
    assert error in events[-1].error.error_message
    await provider.aclose()


async def test_valid_provider_response_completes() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "response",
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = CLIProxyProvider("http://proxy/v1", "key", client=client)
    events = [
        event
        async for event in provider.stream_response(
            model="model",
            system="system",
            messages=[],
            tools=[],
        )
    ]
    assert events[-1].reason == "stop"
    assert events[-1].message.text == "OK"
    await provider.aclose()
