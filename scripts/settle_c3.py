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
    ap.add_argument("--max-samples", type=int, default=2400,
                    help="ceiling when escalating an unconverged run")
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
           f"{'ratio':>7s} {'disp %':>8s} {'x sig':>6s} {'b/e':>6s} {'s/rob':>6s} "
           f"{'kind':>6s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, con, prov in cases:
        # A run that has not converged has decided nothing, so escalate the
        # sample count until it settles rather than quoting a drifting number.
        n = args.samples
        res = None
        try:
            while True:
                res = run_nuisance_budget(
                    pair, con, cfg, ranges=NuisanceRanges(), ljf_model=LJFModel.SHELL,
                    n_samples=n, record_days=14.0, n_theta=12, seed=args.seed,
                    era5_available=False, signature_fraction=CLAIMED,
                )
                converged = res.convergence.converged if res.convergence else False
                # Escalating cannot fix an undefined variance, so stop and use
                # the robust scale instead of burning samples on it.
                if converged or res.heavy_tailed or n >= args.max_samples:
                    break
                n = min(n * 2, args.max_samples)
                print(f"  ... {label[:34]} not converged, retrying at {n} samples")
        except Exception as exc:  # noqa: BLE001
            print(f"{label[:34]:34s} {prov:9s} FAILED: {type(exc).__name__}: {exc}")
            continue
        if res is None:
            continue

        status, _msg = verdict_against_claimed_signature(res, CLAIMED)
        i = con.index("M2")
        maj = float(con.semi_major[i])
        ecc = abs(float(con.semi_minor[i]) / maj) if maj else float("nan")
        be = res.break_even.factor if res.break_even else float("nan")
        conv = res.convergence.converged if res.convergence else False
        rob = res.joint_robust_sd
        s_over_r = res.joint_sd / rob if rob and np.isfinite(rob) and rob > 0 else float("nan")
        kind = "robust" if res.heavy_tailed else "sigma"
        print(
            f"{label[:34]:34s} {prov:9s} {maj:7.3f} {ecc:6.3f} {con.form_factor():6.3f} "
            f"{res.baseline_ratio:7.3f} {res.effective_cv * 100:8.2f} "
            f"{res.effective_cv / CLAIMED:6.2f} {be:6.2f} {s_over_r:6.2f} "
            f"{kind:>6s}  {status}"
        )
        # A heavy-tailed run is decided, not incomplete, so it does not count
        # against the settlement.
        rows.append((label, prov, res, status, conv or res.heavy_tailed, maj, ecc))

    print()
    measured = [r for r in rows if r[1] == "MEASURED"]
    if measured:
        cvs = np.array([r[2].effective_cv for r in measured])
        heavy = [r[0] for r in measured if r[2].heavy_tailed]
        if heavy:
            print(f"  variance UNDEFINED at {len(heavy)} site(s): {'; '.join(heavy)}")
            print("    The strain ratio's denominator passes near zero when a reversing")
            print("    current slackens, so the ratio is Cauchy-like and its variance does")
            print("    not exist. A robust scale is used for those rows. That the method's")
            print("    own statistic has no variance at such a site is a finding against it.")
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

        if verdicts != {"FAIL"}:
            print(
                "\n  NOT SETTLED: the verdict is not the same at every regime "
                f"({', '.join(sorted(verdicts))}). Site-specific constants are required, "
                "because which regime the platform sits in decides the answer."
            )
        elif unconverged:
            print(
                f"\n  CONSISTENT BUT INCOMPLETE: every regime gives FAIL, but "
                f"{len(unconverged)} run(s) had not converged at the sample count used "
                f"({'; '.join(unconverged)}). The direction is not in doubt; the exact sigma "
                "for those rows is. Re-run with a higher --max-samples before quoting them."
            )
        else:
            print(
                "\n  SETTLED: every real tidal regime tested gives the same verdict, and every\n"
                "  run converged. C3's FAIL is not an artefact of the placeholder tide.\n"
                "  Obtaining site-specific TPXO constants would change the numbers but not\n"
                "  the conclusion."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
