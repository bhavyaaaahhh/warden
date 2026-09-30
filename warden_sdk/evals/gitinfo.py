"""Git facts about the code under test, and picking a baseline run from git history.

Every run records the commit it ran on. `check --baseline git` then compares
against the run made on the commit your branch forked from (the merge-base
with the default branch), or the nearest run before it. That's the right
default for "did my branch make things worse than main?".

If git isn't available, or no run matches, it fails loudly instead of quietly
comparing against some other run.
"""

import subprocess
from typing import Any

# How far back from the merge-base to look for a run.
MAX_ANCESTORS = 1000


class GitBaselineError(Exception):
    pass


def _git(*args: str, cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(cwd: str | None = None) -> dict[str, Any] | None:
    """The current commit, branch, and whether there are uncommitted changes. None outside a repo."""
    commit = _git("rev-parse", "HEAD", cwd=cwd)
    if commit is None:
        return None
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd),
        # Uncommitted changes mean the commit doesn't fully describe the code that ran.
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", cwd=cwd)),
    }


def default_branch(cwd: str | None = None) -> str | None:
    remote_head = _git("symbolic-ref", "--short", "refs/remotes/origin/HEAD", cwd=cwd)
    if remote_head:
        return remote_head  # e.g. origin/main
    for name in ("main", "master"):
        if _git("rev-parse", "--verify", "--quiet", name, cwd=cwd):
            return name
    return None


def baseline_commits(cwd: str | None = None) -> tuple[str, list[str]]:
    """The merge-base with the default branch, and its ancestors nearest first."""
    if git_info(cwd) is None:
        raise GitBaselineError("not in a git repository, so there's no history to pick a baseline from")
    branch = default_branch(cwd)
    if branch is None:
        raise GitBaselineError("can't find the default branch (no origin/HEAD, main or master)")
    base = _git("merge-base", "HEAD", branch, cwd=cwd)
    if base is None:
        raise GitBaselineError(f"HEAD has no common history with {branch}")
    ancestors = (_git("rev-list", f"--max-count={MAX_ANCESTORS}", base, cwd=cwd) or "").split()
    return base, ancestors


def pick_git_baseline(runs: list[dict[str, Any]], ancestors: list[str]) -> dict[str, Any] | None:
    """The completed run on the nearest ancestor commit. Runs with uncommitted changes don't count."""
    position = {commit: i for i, commit in enumerate(ancestors)}
    candidates = [
        r for r in runs
        if r.get("status") == "completed"
        and not ((r.get("metadata") or {}).get("git") or {}).get("dirty", True)
        and ((r.get("metadata") or {}).get("git") or {}).get("commit") in position
    ]
    if not candidates:
        return None
    # Nearest commit first; among runs on the same commit, the latest.
    return min(candidates, key=lambda r: (position[r["metadata"]["git"]["commit"]], -_timestamp(r)))


def _timestamp(run: dict[str, Any]) -> float:
    from datetime import datetime

    started = run.get("started_at")
    return datetime.fromisoformat(started).timestamp() if isinstance(started, str) else 0.0
