"""ClaudeModel through the real Anthropic SDK, against a mocked HTTP transport."""

import json

import anthropic
import httpx2
import pytest

from warden_sdk.evals.models import ClaudeModel, ModelError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def _model(respond):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return respond(request)

    client = anthropic.Anthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeModel(client=client), sent


def _message(content, stop_reason="end_turn", stop_details=None):
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def test_sends_a_json_schema_and_parses_the_reply():
    model, sent = _model(lambda r: _message([{"type": "text", "text": '{"ok": true}'}]))
    assert model.json("be a judge", [{"role": "user", "content": "grade this"}], SCHEMA) == {"ok": True}
    [body] = sent
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "be a judge"
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    # Forced tool choice is a 400 on this model; the schema comes through output_config instead.
    assert "tool_choice" not in body and "thinking" not in body
    assert model.name == "claude-opus-5-5@low"


def test_refusal_is_a_model_error():
    model, _ = _model(lambda r: _message([], "refusal", {"type": "refusal", "category": "cyber", "explanation": "no"}))
    with pytest.raises(ModelError, match="refused \\(cyber\\)"):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)


def test_truncated_and_invalid_replies_are_model_errors():
    model, _ = _model(lambda r: _message([{"type": "text", "text": '{"ok": tr'}], "max_tokens"))
    with pytest.raises(ModelError, match="cut off"):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)
    model, _ = _model(lambda r: _message([{"type": "text", "text": "not json"}]))
    with pytest.raises(ModelError, match="not valid JSON"):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)


def test_rate_limits_and_server_errors_are_model_errors_but_bad_requests_raise():
    error = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    model, _ = _model(lambda r: httpx2.Response(429, json=error))
    with pytest.raises(ModelError, match="RateLimitError"):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)
    model, _ = _model(lambda r: httpx2.Response(529, json={**error, "error": {"type": "overloaded_error", "message": "busy"}}))
    with pytest.raises(ModelError, match="API error 529"):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)
    model, _ = _model(lambda r: httpx2.Response(400, json={**error, "error": {"type": "invalid_request_error", "message": "bad"}}))
    with pytest.raises(anthropic.BadRequestError):
        model.json("s", [{"role": "user", "content": "x"}], SCHEMA)


def test_missing_credentials_stop_the_run_with_a_clear_message(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    client = anthropic.Anthropic(api_key=None, http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(500))))
    with pytest.raises(SystemExit, match="no Anthropic credentials"):
        ClaudeModel(client=client).json("s", [{"role": "user", "content": "x"}], SCHEMA)
