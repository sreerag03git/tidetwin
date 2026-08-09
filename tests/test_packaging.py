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
    for name, version in _pins():
        assert version, f"'{name}' is not pinned to an exact version"


def _pins() -> list[tuple[str, str]]:
    import re

    out = []
    for line in (ROOT / "requirements.txt").read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)", line)
        assert m, f"'{line}' is not an exact pin"
        out.append((m.group(1), m.group(2)))
    return out


def _runtime_python() -> tuple[int, int]:
    text = (ROOT / "runtime.txt").read_text("utf-8").strip()
    assert text.startswith("python-"), f"runtime.txt should read 'python-X.Y', got '{text}'"
    major, minor = text.removeprefix("python-").split(".")[:2]
    return int(major), int(minor)


def test_runtime_python_is_declared_consistently():
    """runtime.txt, pyproject and CI must agree, or the build resolves differently
    from anything that was tested."""
    py = _runtime_python()
    declared = f"{py[0]}.{py[1]}"

    pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
    assert f'requires-python = ">={declared}"' in pyproject, (
        f"pyproject.toml does not require >={declared}"
    )

    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    for line in ci.splitlines():
        if "python-version:" in line and not line.strip().startswith("#"):
            assert declared in line, (
                f"CI pins {line.strip()} but runtime.txt says {declared}"
            )


def test_every_pin_supports_the_declared_python():
    """The bug this guards: numpy==2.5.1 needs Python >= 3.12 while runtime.txt
    said python-3.11, so Streamlit Cloud could not resolve the requirements at
    all and the deploy failed before the app ever started.

    Uses the metadata of the installed distributions, so it needs no network. A
    pin whose exact version is not installed here is skipped, and the test fails
    if that leaves it checking nothing.
    """
    from importlib.metadata import PackageNotFoundError, distribution
    from packaging.specifiers import SpecifierSet

    py = _runtime_python()
    version_str = f"{py[0]}.{py[1]}.0"
    checked = 0
    for name, pinned in _pins():
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        if dist.version != pinned:
            continue
        requires = dist.metadata.get("Requires-Python")
        checked += 1
        if not requires:
            continue
        assert SpecifierSet(requires).contains(version_str), (
            f"{name}=={pinned} requires Python {requires}, which excludes the "
            f"{'.'.join(map(str, py))} declared in runtime.txt"
        )
    assert checked >= 3, (
        f"only {checked} pinned packages were installed here, so this test proved "
        "almost nothing; install requirements.txt to make it meaningful"
    )


def test_the_app_ui_module_imports_on_this_platform():
    """The deployed app failed at 'from tidetwin.ui import ...' with a redacted
    ImportError, and nothing else here imported that module - so a real Linux
    import break would have passed CI. Importing it on the CI OS (which is the
    deploy OS) closes that gap: if streamlit, plotly, the components bridge or the
    module's own theme construction cannot load here, this is where it shows.
    """
    import importlib

    ui = importlib.import_module("tidetwin.ui")
    # The theme f-strings are built at import time; make sure they materialised.
    assert "<style>" in ui._CSS
    assert callable(ui.running_reporter)


def test_every_name_app_imports_from_ui_actually_exists():
    """Guards against 'cannot import name X from tidetwin.ui' - the exact shape of
    the deployed failure - without running the Streamlit app (which cannot run
    outside a Streamlit runtime). Parses app.py's import block and checks each
    name against the module, so a rename in ui.py that app.py has not caught up
    with fails here rather than on Cloud.
    """
    import ast
    import importlib

    ui = importlib.import_module("tidetwin.ui")
    tree = ast.parse((ROOT / "app.py").read_text("utf-8"))
    wanted: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tidetwin.ui":
            wanted.extend(alias.name for alias in node.names)
    assert wanted, "app.py should import from tidetwin.ui"
    missing = [n for n in wanted if not hasattr(ui, n)]
    assert not missing, f"app.py imports names tidetwin.ui does not export: {missing}"
