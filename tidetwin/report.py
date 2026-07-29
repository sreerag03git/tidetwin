"""Full report generation.

A report is produced for **any** input and **any** state of data availability.
That is a hard requirement, not a convenience: an adversarial test bench that
falls over when handed unusual inputs cannot be trusted to report honestly on
ordinary ones either. So this module renders whatever exists, states plainly what
does not, and never omits a claim - a claim that could not be evaluated appears
in the report saying exactly that.

Three formats, all self-contained:

``markdown``  for pasting into a paper draft or an issue
``html``      for circulation, styled to match the app, no external assets
``text``      for a terminal or a CI log
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass


from .claims.ledger import Stamp
from .claims.registry import CLAIMS, Artifacts, ClaimResult, Status

__all__ = ["ReportInputs", "to_markdown", "to_html", "to_text"]


@dataclass
class ReportInputs:
    """The configuration a report describes, flattened to displayable rows."""

    rows: list[tuple[str, str]]

    @classmethod
    def from_config(cls, cfg) -> "ReportInputs":
        r: list[tuple[str, str]] = [
            ("Platform latitude", f"{cfg.latitude:.4f} deg N"),
            ("Platform longitude", f"{cfg.longitude:.4f} deg E"),
            ("Target joint", f"J{cfg.joint_id}"),
            ("Gauge offset", f"{cfg.sensor_offset_m:.2f} m"),
            ("Gauge circumferential angle", f"{cfg.sensor_theta_deg:.0f} deg"),
            ("Local joint flexibility", cfg.ljf_model.value),
            ("Member roughness", f"{cfg.roughness_m * 1e3:.0f} mm"),
            ("Marine growth", f"{cfg.marine_growth_mm:.0f} mm"),
            ("Crack a/T", f"{cfg.crack_a_over_T:.3f}"),
            ("Crack 2c", f"{cfg.crack_2c_m * 1e3:.0f} mm"),
            ("Record length", f"{cfg.record_days:.1f} d"),
            ("Sample interval", f"{cfg.sample_interval_s:.0f} s"),
            ("Monte Carlo samples per channel", str(cfg.n_mc_samples)),
            ("Response-surface directions", str(cfg.n_theta)),
            ("Random seed", str(cfg.seed)),
            ("Tidal constants", cfg.tide_source or "labelled placeholder set"),
        ]
        return cls(r)


def _summary_counts(results: list[ClaimResult]) -> dict[str, int]:
    out = {s.value: 0 for s in Status}
    for r in results:
        out[r.status.value] += 1
    return out


def _pending_note(results: list[ClaimResult]) -> str:
    """Say so, prominently, if the report describes an unfinished analysis."""
    n = sum(r.status is Status.NOT_RUN for r in results)
    if not n:
        return ""
    return (
        f"**{n} of {len(results)} claims had not been computed when this report was "
        "generated.** They are marked NOT RUN YET, which is not a verdict and must not be "
        "read as one. Re-run the analysis for a complete ledger."
    )


def _headline(art: Artifacts, results: list[ClaimResult]) -> tuple[str, str]:
    """The C3 verdict, which leads every report regardless of tab or format."""
    c3 = next(r for r in results if r.claim_id == "C3")
    return f"C3 - nuisance variance budget: {c3.status.value}", c3.detail


# ------------------------------------------------------------------ markdown


def to_markdown(
    results: list[ClaimResult],
    art: Artifacts,
    stamp: Stamp,
    inputs: ReportInputs | None = None,
) -> str:
    L: list[str] = []
    a = L.append

    a("# TideTwin claims report")
    a("")
    a(
        "Adversarial evaluation of the claims in *Probabilistic Fatigue Digital Twin for "
        "Offshore Jackets: EnKF with Tidal Calibration Signal*."
    )
    a("")

    pending = _pending_note(results)
    if pending:
        a(f"> {pending}")
        a("")

    title, detail = _headline(art, results)
    a(f"## {title}")
    a("")
    a(f"> {detail}")
    a("")

    counts = _summary_counts(results)
    a("## Summary")
    a("")
    a("| Outcome | Count |")
    a("|---|---|")
    for k, v in counts.items():
        if v:
            a(f"| {k} | {v} |")
    a("")

    a("## Claims ledger")
    a("")
    a("| Claim | Asserted | Computed | Status |")
    a("|---|---|---|---|")
    by_id = {c.id: c for c in CLAIMS}
    for r in results:
        c = by_id[r.claim_id]
        a(f"| **{c.id}** | {c.claimed_value} | {r.computed_text} | **{r.status.value}** |")
    a("")

    a("## Claim detail")
    a("")
    for r in results:
        c = by_id[r.claim_id]
        a(f"### {c.id} — {r.status.value}")
        a("")
        a(f"**Statement.** {c.statement}")
        a("")
        a(f"**Asserted.** {c.claimed_value}  ")
        a(f"**Computed.** {r.computed_text}")
        a("")
        a(f"**Pass criterion.** {c.pass_criterion}")
        a("")
        a(r.detail)
        a("")
        if r.blocking_assumptions:
            a("**Blocking assumptions.**")
            a("")
            for b in dict.fromkeys(r.blocking_assumptions):
                a(f"- {b}")
            a("")

    if art.input_notes:
        a("## Input adjustments")
        a("")
        a("The following inputs were outside the range the solvers can evaluate and were "
          "adjusted. Nothing was corrected silently.")
        a("")
        for n in art.input_notes:
            a(f"- {n}")
        a("")

    if art.errors:
        a("## Computations that did not complete")
        a("")
        for cid, why in sorted(art.errors.items()):
            a(f"- **{cid}** — {why}")
        a("")

    a("## Data source status")
    a("")
    a("| Source | Status |")
    a("|---|---|")
    for label, ok in (
        ("OC4 jacket geometry", True),
        ("Tide model (TPXO/FES)", art.tide_model_available),
        ("ERA5 reanalysis", art.era5_available),
        ("Shell-FE crack-to-LJF surface", art.shell_fe_available),
        ("DNV-RP-C203 S-N curve T", art.sn_available),
        ("BS 7910 Paris constants", art.paris_available),
        ("Published competitor POD curves", art.competitor_pod_available),
    ):
        a(f"| {label} | {'AVAILABLE' if ok else 'UNAVAILABLE'} |")
    a("")

    if art.modelling_assumptions:
        a("## Modelling assumptions in force")
        a("")
        for m in art.modelling_assumptions:
            a(f"- {m}")
        a("")

    if inputs is not None:
        a("## Configuration")
        a("")
        a("| Input | Value |")
        a("|---|---|")
        for k, v in inputs.rows:
            a(f"| {k} | {v} |")
        a("")

    a("## Reproducibility stamp")
    a("")
    a("| Key | Value |")
    a("|---|---|")
    for k, v in stamp.as_rows():
        a(f"| {k} | `{v}` |")
    a("")
    a(
        "*Every number in this report is computed at runtime from a solver, a published "
        "constant or a dataset. Values marked ASSUMED are user inputs and contaminate "
        "everything downstream of them.*"
    )
    a("")
    return "\n".join(L)


# ---------------------------------------------------------------------- text


def to_text(results: list[ClaimResult], art: Artifacts, stamp: Stamp) -> str:
    L: list[str] = []
    a = L.append
    a("=" * 78)
    a("TIDETWIN CLAIMS REPORT")
    a("=" * 78)
    pending = _pending_note(results)
    if pending:
        a("")
        for line in _wrap(pending.replace("**", ""), 78):
            a("  " + line)
    title, detail = _headline(art, results)
    a("")
    a(title.upper())
    for line in _wrap(detail, 78):
        a("  " + line)
    a("")
    a("-" * 78)
    a(f"{'ID':<5}{'STATUS':<38}COMPUTED")
    a("-" * 78)
    by_id = {c.id: c for c in CLAIMS}
    for r in results:
        a(f"{r.claim_id:<5}{r.status.value:<38}{r.computed_text}")
    a("-" * 78)
    counts = _summary_counts(results)
    a("  " + "   ".join(f"{k} {v}" for k, v in counts.items() if v))
    a("")
    for r in results:
        c = by_id[r.claim_id]
        a(f"{c.id} — {r.status.value}")
        for line in _wrap(r.detail, 74):
            a("    " + line)
        a("")
    if art.errors:
        a("COMPUTATIONS THAT DID NOT COMPLETE")
        for cid, why in sorted(art.errors.items()):
            a(f"  {cid}: {why}")
        a("")
    if art.input_notes:
        a("INPUT ADJUSTMENTS")
        for n in art.input_notes:
            for line in _wrap(n, 74):
                a("    " + line)
        a("")
    a("REPRODUCIBILITY STAMP")
    for k, v in stamp.as_rows():
        a(f"  {k:<28}{v}")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# ---------------------------------------------------------------------- html

_HTML_CSS = """
:root{--bg:#ffffff;--panel:#f7f8fa;--line:#e1e5e9;--ink:#14181c;--dim:#5b646d;
 --accent:#1a5fb4;
 --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
 --mono:"SF Mono",Menlo,Consolas,"Liberation Mono","Courier New",monospace}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.25rem;background:var(--bg);color:var(--ink);
 font:16px/1.6 var(--sans);font-feature-settings:"tnum" 1,"lnum" 1}
.wrap{max-width:54rem;margin:0 auto}
h1{font-size:1.75rem;margin:0 0 .3rem;letter-spacing:-.02em;font-weight:650}
h2{font-size:1.25rem;margin:2.3rem 0 .7rem;padding-bottom:.35rem;
 border-bottom:1px solid var(--line);font-weight:640}
h3{font-size:1.02rem;margin:1.5rem 0 .4rem;font-weight:640}
p{margin:.5rem 0}
a{color:var(--accent)}
.sub{color:var(--dim);font-size:.95rem;margin-bottom:1.6rem}
code,.num,td.num{font-family:var(--mono);
 font-variant-numeric:tabular-nums lining-nums;font-size:.9em}
table{width:100%;border-collapse:collapse;margin:.6rem 0 1rem;font-size:.92rem}
th,td{text-align:left;padding:.5rem .65rem;border-bottom:1px solid var(--line);
 vertical-align:top}
th{color:var(--dim);font-weight:700;font-size:.74rem;letter-spacing:.05em;
 text-transform:uppercase}
tbody tr:nth-child(even){background:var(--panel)}
.chip{font-family:var(--sans);font-size:.7rem;letter-spacing:.05em;font-weight:700;
 padding:.16rem .5rem;border:1px solid currentColor;border-radius:3px;
 display:inline-block;white-space:nowrap;background:#fff;text-transform:uppercase}
.verdict{border:1px solid var(--line);border-left-width:5px;background:var(--panel);
 padding:1.1rem 1.3rem;margin:1rem 0 1.8rem;border-radius:4px}
.verdict h2{border:0;margin:0 0 .5rem;font-size:1.12rem}
.claim{border:1px solid var(--line);background:#fff;padding:1rem 1.2rem;margin:.8rem 0;
 border-radius:4px}
.claim .stmt{color:var(--dim);font-size:.93rem}
ul{margin:.4rem 0 .4rem 1.2rem;padding:0}
li{margin:.28rem 0;font-size:.93rem}
footer{margin-top:2.6rem;padding-top:1rem;border-top:1px solid var(--line);
 color:var(--dim);font-size:.85rem}
@media (max-width:640px){body{padding:1.25rem .8rem;font-size:15px}
 table{font-size:.85rem}th,td{padding:.4rem .45rem}}
@media print{body{padding:0}.claim,.verdict{break-inside:avoid}}
"""


def _esc(s) -> str:
    return _html.escape(str(s))


def to_html(
    results: list[ClaimResult],
    art: Artifacts,
    stamp: Stamp,
    inputs: ReportInputs | None = None,
) -> str:
    by_id = {c.id: c for c in CLAIMS}
    title, detail = _headline(art, results)
    c3 = next(r for r in results if r.claim_id == "C3")

    P: list[str] = []
    a = P.append
    a("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>TideTwin claims report</title>")
    a(f"<style>{_HTML_CSS}</style></head><body><div class='wrap'>")
    a("<h1>TideTwin claims report</h1>")
    a(
        "<p class='sub'>Adversarial evaluation of the claims in "
        "<em>Probabilistic Fatigue Digital Twin for Offshore Jackets: EnKF with Tidal "
        "Calibration Signal</em>.</p>"
    )

    pending = _pending_note(results)
    if pending:
        a("<div class='verdict' style='border-left-color:#8a5300'><p>"
          + _esc(pending.replace('**','')) + '</p></div>')

    a(f"<div class='verdict' style='border-left-color:{c3.status.colour}'>")
    a(f"<h2 style='color:{c3.status.colour}'>{_esc(title)}</h2>")
    a(f"<p>{_esc(detail)}</p></div>")

    a("<h2>Claims ledger</h2><table><thead><tr><th>Claim</th><th>Asserted</th>"
      "<th>Computed</th><th>Status</th></tr></thead><tbody>")
    for r in results:
        c = by_id[r.claim_id]
        a(
            f"<tr><td><b>{_esc(c.id)}</b></td><td>{_esc(c.claimed_value)}</td>"
            f"<td class='num'>{_esc(r.computed_text)}</td>"
            f"<td><span class='chip' style='color:{r.status.colour}'>"
            f"{_esc(r.status.value)}</span></td></tr>"
        )
    a("</tbody></table>")

    a("<h2>Claim detail</h2>")
    for r in results:
        c = by_id[r.claim_id]
        a("<div class='claim'>")
        a(
            f"<h3>{_esc(c.id)} &nbsp;<span class='chip' style='color:{r.status.colour}'>"
            f"{_esc(r.status.value)}</span></h3>"
        )
        a(f"<p class='stmt'>{_esc(c.statement)}</p>")
        a(
            f"<p><b>Asserted</b> <span class='num'>{_esc(c.claimed_value)}</span> &nbsp;&middot;&nbsp; "
            f"<b>Computed</b> <span class='num'>{_esc(r.computed_text)}</span></p>"
        )
        a(f"<p>{_esc(r.detail)}</p>")
        if r.blocking_assumptions:
            a("<p><b>Blocking assumptions</b></p><ul>")
            for b in dict.fromkeys(r.blocking_assumptions):
                a(f"<li>{_esc(b)}</li>")
            a("</ul>")
        a("</div>")

    if art.input_notes:
        a("<h2>Input adjustments</h2>")
        a("<p>These inputs were outside the range the solvers can evaluate and were adjusted. "
          "Nothing was corrected silently.</p><ul>")
        for n in art.input_notes:
            a(f"<li>{_esc(n)}</li>")
        a("</ul>")

    if art.errors:
        a("<h2>Computations that did not complete</h2><ul>")
        for cid, why in sorted(art.errors.items()):
            a(f"<li><b>{_esc(cid)}</b> &mdash; {_esc(why)}</li>")
        a("</ul>")

    a("<h2>Data source status</h2><table><thead><tr><th>Source</th><th>Status</th>"
      "</tr></thead><tbody>")
    for label, ok in (
        ("OC4 jacket geometry", True),
        ("Tide model (TPXO/FES)", art.tide_model_available),
        ("ERA5 reanalysis", art.era5_available),
        ("Shell-FE crack-to-LJF surface", art.shell_fe_available),
        ("DNV-RP-C203 S-N curve T", art.sn_available),
        ("BS 7910 Paris constants", art.paris_available),
        ("Published competitor POD curves", art.competitor_pod_available),
    ):
        col = "#2f9e5f" if ok else "#d1495b"
        a(
            f"<tr><td>{_esc(label)}</td><td><span class='chip' style='color:{col}'>"
            f"{'AVAILABLE' if ok else 'UNAVAILABLE'}</span></td></tr>"
        )
    a("</tbody></table>")

    if inputs is not None:
        a("<h2>Configuration</h2><table><thead><tr><th>Input</th><th>Value</th></tr>"
          "</thead><tbody>")
        for k, v in inputs.rows:
            a(f"<tr><td>{_esc(k)}</td><td class='num'>{_esc(v)}</td></tr>")
        a("</tbody></table>")

    a("<h2>Reproducibility stamp</h2><table><tbody>")
    for k, v in stamp.as_rows():
        a(f"<tr><td>{_esc(k)}</td><td class='num'>{_esc(v)}</td></tr>")
    a("</tbody></table>")

    a(
        "<footer>Every number here is computed at runtime from a solver, a published constant "
        "or a dataset. Values marked ASSUMED are user inputs and contaminate everything "
        "downstream of them.</footer>"
    )
    a("</div></body></html>")
    return "\n".join(P)
