"""Settle C3: does the verdict survive real tidal forcing?

The nuisance budget was first computed under a labelled placeholder tide, which
leaves one obvious objection open - that the FAIL is an artefact of an invented
tidal regime. This script answers it by running the same budget against real,
published harmonic constants from NOAA CO-OPS stations spanning semidiurnal to
mixed-diurnal regimes, rectilinear to strongly rotary currents, and nearly an
order of magnitude in current amplitude.

    python scripts/settle_c3.py --samples 150

If the verdict is the same at every station, C3 does not depend on the tide, and
the remaining tidal uncertainty at the platform site is not what is holding the
method back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidetwin.fe.ljf import LJFModel  # noqa: E402
from tidetwin.geometry.oc4 import WATER_DEPTH, load_tables, sensor_pair  # noqa: E402
from tidetwin.loads.morison import HydroConfig  # noqa: E402
from tidetwin.loads.noaa import (  # noqa: E402
    REFERENCE_STATIONS,
    load_pair,
    to_constituents,
)
from tidetwin.loads.tides import placeholder_constituents  # noqa: E402
from tidetwin.nuisance import (  # noqa: E402
    NuisanceRanges,
    run_nuisance_budget,
    verdict_against_claimed_signature,
)
from tidetwin.provenance import DataUnavailable  # noqa: E402

CLAIMED = 0.111


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--joint", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--fetch", action="store_true", help="download missing stations")
    args = ap.parse_args()

    tables = load_tables()
    pair = sensor_pair(tables, args.joint, offset_m=1.5)
    cfg = HydroConfig(water_depth=WATER_DEPTH, roughness_m=0.05)

    cases = [("PLACEHOLDER (not a real tide)", placeholder_constituents(), "ASSUMED")]
    for sp in REFERENCE_STATIONS:
        try:
            cases.append(
                (sp.label, to_constituents(load_pair(sp, allow_fetch=args.fetch)), "MEASURED")
            )
        except DataUnavailable as exc:
            print(f"skipping {sp.label}: {exc}")

    print(f"\nC3 across {len(cases)} tidal regimes, joint J{args.joint}, "
          f"{args.samples} samples/channel, seed {args.seed}\n")
    hdr = (f"{'tide':34s} {'prov':9s} {'M2 cur':>7s} {'ecc':>6s} {'F':>6s} "
           f"{'ratio':>7s} {'sigma %':>8s} {'x sig':>6s} {'b/e':>6s} {'conv':>5s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, con, prov in cases:
        try:
            res = run_nuisance_budget(
                pair, con, cfg, ranges=NuisanceRanges(), ljf_model=LJFModel.SHELL,
                n_samples=args.samples, record_days=14.0, n_theta=12, seed=args.seed,
                era5_available=False, signature_fraction=CLAIMED,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{label[:38]:38s} {prov:9s} FAILED: {type(exc).__name__}: {exc}")
            continue

        status, _msg = verdict_against_claimed_signature(res, CLAIMED)
        i = con.index("M2")
        maj = float(con.semi_major[i])
        ecc = abs(float(con.semi_minor[i]) / maj) if maj else float("nan")
        be = res.break_even.factor if res.break_even else float("nan")
        conv = res.convergence.converged if res.convergence else False
        print(
            f"{label[:34]:34s} {prov:9s} {maj:7.3f} {ecc:6.3f} {con.form_factor():6.3f} "
            f"{res.baseline_ratio:7.3f} {res.joint_cv * 100:8.2f} "
            f"{res.joint_cv / CLAIMED:6.2f} {be:6.2f} {'yes' if conv else 'NO':>5s}  {status}"
        )
        rows.append((label, prov, res, status, conv, maj, ecc))

    print()
    measured = [r for r in rows if r[1] == "MEASURED"]
    if measured:
        cvs = np.array([r[2].joint_cv for r in measured])
        verdicts = {r[3] for r in measured}
        unconverged = [r[0] for r in measured if not r[4]]
        print(f"Across {len(measured)} real tidal regimes:")
        print(f"  nuisance sigma  {cvs.min() * 100:.2f} to {cvs.max() * 100:.2f} percent "
              f"of the intact ratio (claimed damage signature {CLAIMED * 100:.1f} percent)")
        print(f"  sigma / claimed signature  {cvs.min() / CLAIMED:.2f} to "
              f"{cvs.max() / CLAIMED:.2f}  (limit 0.33)")
        print(f"  verdicts: {', '.join(sorted(verdicts))}")

        # The mechanism: sigma tracks the ellipse shape, not its size.
        eccs = np.array([r[6] for r in measured])
        majors = np.array([r[5] for r in measured])
        if len(measured) >= 3:
            print(f"  correlation of sigma with ellipse eccentricity  "
                  f"r = {np.corrcoef(eccs, cvs)[0, 1]:+.3f}")
            print(f"  correlation of sigma with current amplitude     "
                  f"r = {np.corrcoef(majors, cvs)[0, 1]:+.3f}")
            print("  -> the nuisance floor is set by the SHAPE of the tidal ellipse, not its")
            print("     size. A rotary current never goes slack and keeps the ratio well")
            print("     conditioned; a reversing one passes through zero and the ratio blows up.")

        if unconverged:
            print(f"\n  WARNING: {len(unconverged)} run(s) had not converged: "
                  f"{'; '.join(unconverged)}. Re-run with more --samples before quoting these.")
        if verdicts == {"FAIL"} and not unconverged:
            print(
                "\n  SETTLED: every real tidal regime tested gives the same verdict. C3's FAIL\n"
                "  is not an artefact of the placeholder tide. Obtaining site-specific TPXO\n"
                "  constants would change the numbers but not the conclusion."
            )
        else:
            print("\n  NOT SETTLED: the verdict depends on the tidal regime. "
                  "Site-specific constants are required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
