"""Which commit is this process actually running?

Streamlit Community Cloud deploys by cloning the repo to ``/mount/src/<repo>``
and running the app from there, so the ``.git`` directory is present at runtime.
Reading the checked-out revision straight from ``.git`` lets the app display the
commit it is really serving. That turns "the deploy looks stale" from a guess
into something you can read off the screen: if the badge does not match GitHub's
``main``, Cloud is running a cached older checkout and needs a fresh deploy.

Everything here is pure file reads - no subprocess, no git binary, no heavy
imports - and every path degrades to ``"unknown"`` rather than raising. This
module is imported at app start-up, so a fault in it must never be able to
become the very import failure it exists to diagnose.
"""

from __future__ import annotations

from pathlib import Path

# Repo root is the parent of the package directory (…/repo/tidetwin/buildinfo.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]

_SHORT = 7


def _read(path: Path) -> str | None:
    try:
        return path.read_text("utf-8").strip()
    except OSError:
        return None


def _git_dir(root: Path) -> Path | None:
    """The .git directory, resolving the ``gitdir: …`` indirection used by
    worktrees and submodules. Returns None if there is no git metadata."""
    dot = root / ".git"
    if dot.is_dir():
        return dot
    # A linked worktree stores ".git" as a file: "gitdir: /path/to/real/gitdir".
    text = _read(dot)
    if text and text.startswith("gitdir:"):
        target = Path(text.split(":", 1)[1].strip())
        if not target.is_absolute():
            target = (root / target).resolve()
        return target if target.is_dir() else None
    return None


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve a ref name (e.g. 'refs/heads/main') to a full SHA, checking the
    loose ref file first and then packed-refs."""
    loose = _read(git_dir / ref)
    if loose:
        return loose
    packed = _read(git_dir / "packed-refs")
    if not packed:
        return None
    for line in packed.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def deployed_sha(root: Path | None = None) -> str | None:
    """The full 40-char commit SHA this checkout is on, or None if unknown."""
    git_dir = _git_dir(root or _REPO_ROOT)
    if git_dir is None:
        return None
    head = _read(git_dir / "HEAD")
    if not head:
        return None
    if head.startswith("ref:"):
        return _resolve_ref(git_dir, head.split(":", 1)[1].strip())
    # Detached HEAD: the file holds the SHA directly.
    return head if len(head) >= _SHORT and all(c in "0123456789abcdef" for c in head) else None


def deployed_revision(root: Path | None = None) -> str:
    """Short SHA for display, or 'unknown' when there is no git metadata
    (e.g. running from a source tarball rather than a clone)."""
    sha = deployed_sha(root)
    return sha[:_SHORT] if sha else "unknown"
