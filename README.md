# Warden

Tracing and offline evaluation for LLM agents.

- **Tracing:** `warden_sdk.trace` records an agent's LLM calls, tool calls and retrievals, with tokens, cost and errors, to a Postgres-backed server with a viewer.
- **Offline evals:** run datasets (single calls, scripted conversations, or simulated users) through your agent, score them (checks, tool calls, end state, LLM judges, human labels), and compare runs with a verdict that accounts for run-to-run noise. See [docs/evals.md](docs/evals.md).

```bash
docker compose up -d
uv run alembic upgrade head
uv run uvicorn server.app:app --reload          # viewer at http://localhost:8000
uv run python -m warden_sdk.evals check examples/datasets/support.jsonl --agent examples.support_agent:chat
```

The research and design behind the evals are in [docs/research/](docs/research/).
