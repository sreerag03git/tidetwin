"""Regenerate data/geometry/MANIFEST.json.

Run after any intentional change to the shipped OC4 tables. The app refuses to
load geometry whose hash does not match the manifest, so this script is the only
sanctioned way to change them: the diff on MANIFEST.json makes the edit visible
in review rather than silent.

    python scripts/write_geometry_manifest.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geometry"

SOURCE = {
    "document": (
        "NREL FAST CertTest deck NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat, encoding the "
        "OC4/UpWind reference jacket of Vorpahl, Popko & Kaufer (2013), 'Description of a "
        "basic model of the UpWind reference jacket for code comparison in the OC4 project "
        "under IEA Wind Annex 30', Fraunhofer IWES"
    ),
    "url": (
        "https://github.com/old-NWTC/FAST/blob/master/CertTest/5MW_Baseline/"
        "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
    ),
    "tables": "JOINTS, MEMBERS, MEMBER X-SECTION PROPERTY",
    "retrieved": "2026-07-28",
    "transcription": (
        "Coordinates, connectivity and property sets copied without modification. "
        "The 'role' column in oc4_sections.csv is our own label for readability and "
        "carries no numerical content."
    ),
}


def main() -> None:
    files = {}
    for p in sorted(DATA.glob("oc4_*.csv")):
        data = p.read_bytes()
        files[p.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "rows": len(data.decode("utf-8").strip().splitlines()) - 1,
        }
    manifest = {
        "generated": dt.date.today().isoformat(),
        "source": SOURCE,
        "files": files,
    }
    out = DATA / "MANIFEST.json"
    # newline="" prevents CRLF translation on Windows; the hashes recorded here
    # must describe bytes that are identical on every platform.
    out.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline=""
    )
    print(f"wrote {out}")
    for name, meta in files.items():
        print(f"  {name}: {meta['rows']} rows, sha256 {meta['sha256'][:16]}...")


if __name__ == "__main__":
    main()
