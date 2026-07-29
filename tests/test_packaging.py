"""The repository must work when someone else clones it.

These guard failure modes that never appear in the working tree the code was
written in, and so are invisible to every other test: line-ending translation
changing hash-verified data, deployment files going missing, and CI paths
drifting away from the layout they assume.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_geometry_bytes_match_the_manifest_exactly():
    """The check the app performs at startup, run against the files on disk.

    Fails if git translated line endings on checkout, which is what breaks a
    Windows clone. `.gitattributes` pins these files to no translation.
    """
    manifest = json.loads((ROOT / "data" / "geometry" / "MANIFEST.json").read_text("utf-8"))
    for name, meta in manifest["files"].items():
        data = (ROOT / "data" / "geometry" / name).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        assert got == meta["sha256"], (
            f"{name} hash {got[:12]} != manifest {meta['sha256'][:12]}. If this appeared "
            "after a fresh clone, git translated line endings; check .gitattributes."
        )
        assert len(data) == meta["bytes"]


def test_hash_verified_data_has_no_carriage_returns():
    """CRLF in these files means .gitattributes is not doing its job."""
    targets = list((ROOT / "data" / "geometry").glob("*.csv"))
    targets += list((ROOT / "data" / "constituents").glob("*.json"))
    assert targets, "no data files found to check"
    for p in targets:
        assert b"\r\n" not in p.read_bytes(), f"{p.name} contains CRLF"


def test_gitattributes_pins_the_hashed_files():
    ga = ROOT / ".gitattributes"
    assert ga.is_file(), ".gitattributes is required for reproducible hashes"
    text = ga.read_text("utf-8")
    assert "eol=lf" in text
    for pattern in ("data/geometry/*.csv", "data/constituents/*.json"):
        assert pattern in text, f"{pattern} is not pinned in .gitattributes"


@pytest.mark.parametrize(
    "name",
    ["app.py", "requirements.txt", "runtime.txt", "README.md", "DEPLOY.md", "LICENSE",
     "pyproject.toml", ".streamlit/config.toml", ".github/workflows/ci.yml"],
)
def test_deployment_files_are_present_at_the_root(name):
    """Streamlit Cloud and GitHub Actions both look at the repository root."""
    assert (ROOT / name).is_file(), f"{name} missing from the repository root"


def test_ci_paths_match_this_layout():
    """CI assumes TideTwin is the repository, not a subdirectory of one.

    A stale `tidetwin/` prefix makes the workflow silently match nothing, which
    looks exactly like a passing build.
    """
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert "working-directory: tidetwin" not in ci
    assert "cache-dependency-path: tidetwin/" not in ci
    for line in ci.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "tidetwin/requirements" not in stripped
        assert "tidetwin/ledger" not in stripped


def test_secrets_are_not_committed():
    assert not (ROOT / ".streamlit" / "secrets.toml").exists(), (
        "secrets.toml must never be committed"
    )
    assert "secrets.toml" in (ROOT / ".gitignore").read_text("utf-8")


def test_requirements_are_pinned_exactly():
    """A claims ledger that cannot be reproduced is not evidence."""
    for line in (ROOT / "requirements.txt").read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"'{line}' is not pinned to an exact version"
