"""Run the full analysis headless and emit the claims ledger.

Used by CI to regenerate the README table, and by anyone who wants the numbers
without the UI.

    python scripts/run_ledger.py --out ledger.csv --readme ../README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidetwin.analysis import AnalysisConfig, run_full  # noqa: E402
from tidetwin.claims.ledger import (  # noqa: E402
    build_stamp,
    markdown_summary,
    to_csv,
    to_latex,
)
from tidetwin.claims.registry import CLAIMS, Status, evaluate_all  # noqa: E402
from tidetwin.geometry.oc4 import OC4_CITATION, load_tables  # noqa: E402
from tidetwin.loads.noaa import available_cached  # noqa: E402
from tidetwin.report import ReportInputs, to_html, to_markdown, to_text  # noqa: E402

MARKER_START = "<!-- CLAIMS-LEDGER:START -->"
MARKER_END = "<!-- CLAIMS-LEDGER:END -->"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--latex", type=Path, default=None)
    ap.add_argument("--readme", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="write the full report; format from the suffix (.html, .md, .txt)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--station",
        default="auto",
        help="NOAA current-station id for MEASURED tidal forcing; 'auto' uses the first "
        "cached reference station, 'none' uses the ASSUMED placeholder",
    )
    args = ap.parse_args()

    station = args.station
    if station == "auto":
        cached = available_cached()
        station = cached[0].current_id if cached else None
    elif station == "none":
        station = None

    cfg = AnalysisConfig(
        seed=args.seed, n_mc_samples=args.samples, n_theta=12, record_days=14.0,
        tide_station=station,
    )

    def progress(f: float, m: str) -> None:
        if not args.quiet:
            print(f"  [{f * 100:5.1f}%] {m}", flush=True)

    art = run_full(cfg, progress)
    results = evaluate_all(art)
    tables = load_tables()
    stamp = build_stamp(
        seed=cfg.seed,
        geometry_digest=tables.digest,
        geometry_retrieved=str(OC4_CITATION.retrieved),
        ljf_model=cfg.ljf_model.value,
        tide_source=(
            f"{art.tide_provenance}: {art.tide_source_note}"
            if art.tide_provenance == "MEASURED"
            else f"{art.tide_provenance} placeholder"
        ),
    )

    by_id = {r.claim_id: r for r in results}
    print()
    print(f"{'ID':<4} {'STATUS':<38} COMPUTED")
    print("-" * 100)
    for c in CLAIMS:
        r = by_id[c.id]
        print(f"{c.id:<4} {r.status.value:<38} {r.computed_text}")
    print("-" * 100)
    n_fail = sum(r.status is Status.FAIL for r in results)
    print(
        f"PASS {sum(r.status is Status.PASS for r in results)}   "
        f"MARGINAL {sum(r.status is Status.MARGINAL for r in results)}   "
        f"FAIL {n_fail}   "
        f"UNTESTABLE {sum(r.status.value.startswith('UNTESTABLE') for r in results)}"
    )
    print()
    print("C3 (deciding test):", by_id["C3"].status.value)
    print(" ", by_id["C3"].detail)

    if args.report:
        inputs = ReportInputs.from_config(cfg)
        writer = {
            ".html": lambda: to_html(results, art, stamp, inputs),
            ".md": lambda: to_markdown(results, art, stamp, inputs),
            ".txt": lambda: to_text(results, art, stamp),
        }.get(args.report.suffix.lower())
        if writer is None:
            print(f"unknown report format '{args.report.suffix}'; use .html, .md or .txt")
            return 2
        args.report.write_text(writer(), encoding="utf-8")
        print(f"\nwrote {args.report}")

    if args.out:
        args.out.write_text(to_csv(results, stamp), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.latex:
        args.latex.write_text(to_latex(results, stamp), encoding="utf-8")
        print(f"wrote {args.latex}")
    if args.readme and args.readme.is_file():
        text = args.readme.read_text(encoding="utf-8")
        table = markdown_summary(results, stamp)
        if MARKER_START in text and MARKER_END in text:
            head = text.split(MARKER_START)[0]
            tail = text.split(MARKER_END)[1]
            args.readme.write_text(
                f"{head}{MARKER_START}\n{table}\n{MARKER_END}{tail}", encoding="utf-8"
            )
            print(f"updated the claims table in {args.readme}")
        else:
            print(f"warning: markers not found in {args.readme}; table not written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
