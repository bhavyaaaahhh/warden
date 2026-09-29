import json

import httpx
import pytest


class FakeServer:
    """Records what the runner sends, in place of the Warden server."""

    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.results: dict[str, list[dict]] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        if request.method == "POST" and path == "/eval_runs":
            self.runs[body["run_id"]] = body
            self.results[body["run_id"]] = []
            return httpx.Response(201, json={"run_id": body["run_id"]})
        if request.method == "POST" and path.endswith("/results"):
            self.results[path.split("/")[2]].append(body)
            return httpx.Response(201, json={})
        if request.method == "PATCH":
            self.runs[path.split("/")[2]].update(body)
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": f"fake server: no route {request.method} {path}"})


@pytest.fixture
def fake_server(monkeypatch):
    server = FakeServer()
    real_client = httpx.Client

    def client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(server.handle), **kwargs)

    monkeypatch.setattr("warden_sdk.evals.runner.run.httpx.Client", client)
    return server
