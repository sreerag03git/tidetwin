"""Regenerate the precomputed default bundle the deployed app opens on.

    python scripts/precompute_default.py

Run this whenever any solver source under tidetwin/ changes. The bundle carries
a content fingerprint, so a stale bundle is ignored at load rather than serving a
wrong answer - but ignored means the app computes on demand again, which is the
slow path this bundle exists to avoid. tests/test_precompute.py fails when the
committed bundle is stale, so CI will not let it drift silently.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tidetwin.precompute import load_bundle, save_bundle  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    print("computing the default-config result (claims, sensitivity, tidal cycle)...")
    path, size = save_bundle()
    dt = time.perf_counter() - t0
    print(f"wrote {path}  ({size / 1e6:.2f} MB) in {dt:.1f} s")
    ok = load_bundle() is not None
    print("reload + fingerprint check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
