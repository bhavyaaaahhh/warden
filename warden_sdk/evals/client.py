"""Talking to the Warden server from the CLI."""

from typing import Any
from uuid import UUID

import httpx

from warden_sdk.tracer import WARDEN_URL


def client() -> httpx.Client:
    return httpx.Client(base_url=WARDEN_URL, timeout=10.0)

def fetch_run(client: httpx.Client, ref: str, exclude_run_id: str | None = None) -> dict[str, Any]:
    """Accept either a run id or a version tag (latest completed run with that tag).

    exclude_run_id skips a run when resolving a tag, so a check doesn't pick the
    run it just made as its own baseline.
    """
    try:
        run_id = str(UUID(ref))
    except ValueError:
        runs = client.get(
            "/eval_runs", params={"version_tag": ref, "status": "completed", "limit": 2, "verdicts": False}
        )
        runs.raise_for_status()
        matches = [r["run_id"] for r in runs.json() if r["run_id"] != exclude_run_id]
        if not matches:
            raise SystemExit(f"no completed eval run with version_tag {ref!r}")
        run_id = matches[0]
    res = client.get(f"/eval_runs/{run_id}")
    if res.status_code == 404:
        raise SystemExit(f"eval run {run_id} not found")
    res.raise_for_status()
    return res.json()
