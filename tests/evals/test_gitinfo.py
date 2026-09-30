import subprocess

import pytest

from warden_sdk.evals.gitinfo import (
    GitBaselineError,
    baseline_commits,
    git_info,
    pick_git_baseline,
)


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """main: c1 - c2; feature branched at c2 with c3."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    commits = {}
    for name in ("c1", "c2"):
        (tmp_path / "f").write_text(name)
        git(tmp_path, "add", "f")
        git(tmp_path, "commit", "-qm", name)
        commits[name] = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    (tmp_path / "g").write_text("x")
    git(tmp_path, "add", "g")
    git(tmp_path, "commit", "-qm", "c3")
    commits["c3"] = git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, commits


def run(run_id, commit, dirty=False, started="2026-09-29T10:00:00+00:00", status="completed"):
    return {"run_id": run_id, "status": status, "started_at": started,
            "metadata": {"git": {"commit": commit, "dirty": dirty}}}


def test_git_info(repo):
    path, commits = repo
    info = git_info(str(path))
    assert info == {"commit": commits["c3"], "branch": "feature", "dirty": False}
    (path / "g").write_text("changed")
    assert git_info(str(path))["dirty"] is True


def test_baseline_commits_start_at_the_merge_base(repo):
    path, commits = repo
    base, ancestors = baseline_commits(str(path))
    assert base == commits["c2"]
    assert ancestors == [commits["c2"], commits["c1"]]


def test_picks_the_nearest_ancestor_and_skips_dirty_and_unrelated_runs(repo):
    path, commits = repo
    _, ancestors = baseline_commits(str(path))
    runs = [
        run("on-c1", commits["c1"]),
        run("on-c2-dirty", commits["c2"], dirty=True),
        run("on-feature", commits["c3"]),
        run("elsewhere", "f" * 40),
    ]
    assert pick_git_baseline(runs, ancestors)["run_id"] == "on-c1"
    runs.append(run("on-c2-old", commits["c2"], started="2026-09-28T10:00:00+00:00"))
    runs.append(run("on-c2-new", commits["c2"], started="2026-09-29T10:00:00+00:00"))
    assert pick_git_baseline(runs, ancestors)["run_id"] == "on-c2-new"


def test_no_match_is_none_and_no_repo_is_an_error(tmp_path):
    assert pick_git_baseline([run("x", "a" * 40)], ["b" * 40]) is None
    with pytest.raises(GitBaselineError, match="not in a git repository"):
        baseline_commits(str(tmp_path))
