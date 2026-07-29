"""Ledger export: the stamped table a paper can cite.

The stamp is the point. A claims table without a provenance stamp is an
assertion; with one, a reader can reconstruct exactly which code, which
geometry, which datasets and which random seed produced each verdict.
"""

from __future__ import annotations

import datetime as _dt
import io
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..provenance import Provenance
from .registry import CLAIMS, Artifacts, ClaimResult

__all__ = ["APP_VERSION", "Stamp", "build_stamp", "ledger_frame", "to_csv", "to_latex"]

APP_VERSION = "0.1.0"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


@dataclass(frozen=True)
class Stamp:
    """Reproducibility metadata attached to every export."""

    app_version: str
    git_commit: str
    git_dirty: bool
    generated_utc: str
    python_version: str
    seed: int
    geometry_digest: str
    geometry_retrieved: str
    shell_fe_digest: str
    tide_source: str
    era5_retrieved: str
    ljf_model: str
    extra: dict[str, str] = field(default_factory=dict)

    def as_rows(self) -> list[tuple[str, str]]:
        rows = [
            ("app version", self.app_version),
            ("git commit", self.git_commit + (" (dirty)" if self.git_dirty else "")),
            ("generated (UTC)", self.generated_utc),
            ("python", self.python_version),
            ("random seed", str(self.seed)),
            ("OC4 geometry digest", self.geometry_digest),
            ("OC4 geometry retrieved", self.geometry_retrieved),
            ("shell-FE surface", self.shell_fe_digest),
            ("tidal constants", self.tide_source),
            ("ERA5 retrieved", self.era5_retrieved),
            ("LJF formulation", self.ljf_model),
        ]
        rows.extend(self.extra.items())
        return rows

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_rows(), columns=["key", "value"])


def build_stamp(
    seed: int,
    geometry_digest: str,
    geometry_retrieved: str,
    ljf_model: str,
    shell_fe_digest: str = "not shipped",
    tide_source: str = "ASSUMED placeholder",
    era5_retrieved: str = "not retrieved",
    **extra: str,
) -> Stamp:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return Stamp(
        app_version=APP_VERSION,
        git_commit=commit,
        git_dirty=dirty,
        generated_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python_version=platform.python_version(),
        seed=seed,
        geometry_digest=geometry_digest,
        geometry_retrieved=geometry_retrieved,
        shell_fe_digest=shell_fe_digest,
        tide_source=tide_source,
        era5_retrieved=era5_retrieved,
        ljf_model=ljf_model,
        extra=dict(extra),
    )


def ledger_frame(results: list[ClaimResult]) -> pd.DataFrame:
    """The claims table, one row per claim."""
    by_id = {c.id: c for c in CLAIMS}
    rows = []
    for r in results:
        c = by_id[r.claim_id]
        rows.append(
            {
                "id": c.id,
                "statement": c.statement,
                "claimed": c.claimed_value,
                "computed": r.computed_text,
                "status": r.status.value,
                "pass_criterion": c.pass_criterion,
                "blocking_assumptions": " | ".join(r.blocking_assumptions),
                "detail": r.detail,
            }
        )
    return pd.DataFrame(rows)


def to_csv(results: list[ClaimResult], stamp: Stamp) -> str:
    """CSV with the stamp as a commented preamble."""
    buf = io.StringIO()
    for k, v in stamp.as_rows():
        buf.write(f"# {k}: {v}\n")
    buf.write("#\n")
    ledger_frame(results).to_csv(buf, index=False)
    return buf.getvalue()


def _tex_escape(s: str) -> str:
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(a, b)
    return s


def to_latex(results: list[ClaimResult], stamp: Stamp) -> str:
    """A two-column-friendly LaTeX table, stamped in the caption."""
    by_id = {c.id: c for c in CLAIMS}
    lines = [
        "% TideTwin claims ledger",
        f"% generated {stamp.generated_utc}, commit {stamp.git_commit}, seed {stamp.seed}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\hline",
        r"ID & Claimed & Computed & Status \\",
        r"\hline",
    ]
    for r in results:
        c = by_id[r.claim_id]
        lines.append(
            f"{_tex_escape(c.id)} & {_tex_escape(c.claimed_value)} & "
            f"{_tex_escape(r.computed_text)} & {_tex_escape(r.status.value)} \\\\"
        )
    stamp_txt = (
        f"TideTwin {stamp.app_version}, commit {stamp.git_commit}"
        + (" (dirty working tree)" if stamp.git_dirty else "")
        + f", seed {stamp.seed}, OC4 geometry {stamp.geometry_digest}, "
        f"LJF {stamp.ljf_model}, shell-FE surface {stamp.shell_fe_digest}, "
        f"tidal constants {stamp.tide_source}, generated {stamp.generated_utc}."
    )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{Claims ledger. " + _tex_escape(stamp_txt) + "}",
        r"\label{tab:claims-ledger}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def markdown_summary(results: list[ClaimResult], stamp: Stamp) -> str:
    """Markdown table for the README, regenerated by CI."""
    by_id = {c.id: c for c in CLAIMS}
    out = ["| Claim | Asserted | Computed | Status |", "|---|---|---|---|"]
    for r in results:
        c = by_id[r.claim_id]
        stmt = c.statement if len(c.statement) < 90 else c.statement[:87] + "..."
        out.append(
            f"| **{c.id}** {stmt} | {c.claimed_value} | {r.computed_text} | **{r.status.value}** |"
        )
    out.append("")
    # Deliberately no commit hash and no wall-clock time. This block lives *in*
    # the commit it would name, so the hash could only ever be the previous
    # commit's - it can never be right. Worse, both fields change on every run
    # even when no number does, so CI regenerated and committed the table on
    # every push, and each such commit collided with the next local run.
    #
    # What belongs here is what determines the numbers: the code version, the
    # seed, the geometry the digest pins, the joint model, and the tidal source.
    # If any of those is unchanged the table is unchanged, so CI now commits only
    # when a result has actually moved. The full stamp, commit hash and timestamp
    # included, is still written to ledger.csv and ledger.tex.
    out.append(
        f"<sub>TideTwin {stamp.app_version} - seed {stamp.seed} - "
        f"OC4 geometry `{stamp.geometry_digest}` - LJF {stamp.ljf_model} - "
        f"tidal constants {stamp.tide_source}</sub>"
    )
    return "\n".join(out)
