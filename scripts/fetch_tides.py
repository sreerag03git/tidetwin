"""Fetch real tidal harmonic constants and cache them under data/constituents/.

    python scripts/fetch_tides.py            # all reference stations
    python scripts/fetch_tides.py --list     # show what is already cached

NOAA CO-OPS needs no account and no API key. The cached JSON is committed so the
app has MEASURED tidal forcing out of the box and CI can reproduce the ledger
without network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tidetwin.loads.noaa import (  # noqa: E402
    REFERENCE_STATIONS,
    available_cached,
    fetch_pair,
    load_pair,
    to_constituents,
)
from tidetwin.provenance import DataUnavailable  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show cached stations and exit")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    if args.list:
        cached = available_cached()
        print(f"{len(cached)} of {len(REFERENCE_STATIONS)} reference stations cached")
        for p in REFERENCE_STATIONS:
            mark = "cached" if p in cached else "     -"
            print(f"  [{mark}] {p.label}")
        return 0

    ok = 0
    for pair in REFERENCE_STATIONS:
        try:
            if pair.cache_path.is_file() and not args.force:
                payload = load_pair(pair)
                action = "cached"
            else:
                payload = fetch_pair(pair)
                action = "fetched"
            con = to_constituents(payload)
            m2 = con.semi_major[con.index("M2")]
            ecc = abs(con.semi_minor[con.index("M2")] / m2) if m2 else float("nan")
            print(
                f"[{action}] {pair.label}\n"
                f"          M2 current {m2:.3f} m/s, eccentricity {ecc:.3f}, "
                f"elevation {con.elev_amp[con.index('M2')]:.3f} m, "
                f"form factor F={con.form_factor():.3f}, "
                f"spring/neap {con.spring_neap_ratio():.2f}, "
                f"stations {payload['separation_km']} km apart"
            )
            ok += 1
        except DataUnavailable as exc:
            print(f"[failed ] {pair.label}: {exc}")
    print(f"\n{ok}/{len(REFERENCE_STATIONS)} station pairs available")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
