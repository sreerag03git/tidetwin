# Deploying TideTwin

Exact commands. Nothing here authenticates on your behalf; you run the pushes
and click the dashboard yourself.

---

## 1. Local check before you push

From the `tidetwin/` directory:

```bash
python -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt pytest==9.1.1
```

Windows PowerShell uses `.venv\Scripts\pip` instead.

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

All tests must pass. Then confirm the analysis runs headless and the ledger
builds:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_ledger.py --samples 80 --out ledger.csv --readme README.md
```

Finally, run the app:

```bash
PYTHONPATH=. .venv/bin/streamlit run app.py
```

Cold start should reach the Overview tab in under 15 seconds. The C3 verdict
banner appears at the top of every tab; the expensive Monte Carlo waits behind
**Run full analysis**.

---

## 2. Initialise the repository

TideTwin is self-contained in this directory, and there are two ways to deploy
it. Both work; the first is cleaner.

**Option A — its own repository (recommended).** `.streamlit/config.toml`,
`requirements.txt` and `runtime.txt` then sit at the repository root, exactly
where Streamlit Cloud looks for them. Follow the steps below as written.

**Option B — from the existing monorepo.** If `tidetwin/` stays inside
`scr-fatigue-twin`, set the main file path to `tidetwin/app.py` in the Streamlit
dashboard. `app.py` inserts its own directory on `sys.path`, so the imports
resolve either way. Two things to watch:

- Streamlit Cloud may pick up the **repository root** `requirements.txt`, which
  belongs to the other application in that repo and pins different versions.
  Check the build log; if it installed the wrong set, use Option A.
- The root `.streamlit/config.toml` is a light theme for the other app. TideTwin
  injects its own CSS so it still renders correctly, but the browser chrome
  colour will come from the root config.

```bash
git init -b main
```

```bash
git add .
```

```bash
git commit -m "TideTwin: adversarial test bench for the tidal calibration claims"
```

Confirm the secrets file is not staged — it must never be committed:

```bash
git ls-files --error-unmatch .streamlit/secrets.toml
```

That command should **fail** with "did not match any file". If it succeeds, stop
and remove the file from the index.

---

## 3. First push

Create an empty repository on GitHub (no README, no licence — this directory has
both), then:

```bash
git remote add origin https://github.com/<you>/tidetwin.git
```

```bash
git push -u origin main
```

---

## 4. Connect it at share.streamlit.io

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `<you>/tidetwin`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: **3.11** (matches `runtime.txt`)
4. Open **Advanced settings** before deploying and paste the secrets block from
   step 5.
5. **Deploy**.

The first build installs the pinned requirements and takes a few minutes.
Subsequent pushes to `main` redeploy automatically.

---

## 5. Set the CDS API key in the Streamlit Cloud secrets UI

ERA5 access needs a free Copernicus Climate Data Store account.

1. Register at <https://cds.climate.copernicus.eu>, then open your profile page
   to find your API key.
2. Accept the ERA5 licence — retrievals fail with a licence error until you do.
3. In the Streamlit Cloud dashboard: your app → **⋮** → **Settings** →
   **Secrets**, and paste:

```toml
CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
CDSAPI_KEY = "your-key-here"
```

4. Save. The app restarts automatically.

Without this, the Environment tab shows
`DATA UNAVAILABLE — CDS credentials not configured` with these instructions, and
C5 plus the wind and wave channels of C3 report `UNTESTABLE — DATA MISSING`. The
app is fully usable in that state; it simply reports less.

Never commit `.streamlit/secrets.toml`. It is in `.gitignore`, and step 2
verifies it.

---

## 6. Verify the deployed build

Work through this list on the live URL:

- [ ] The C3 verdict banner renders at the top, above the tabs.
- [ ] **Structure** draws the OC4 jacket and reports 64 joints, 112 members,
      50.001 m water depth.
- [ ] **Provenance** shows the geometry as AVAILABLE and lists the unshipped
      sources as UNAVAILABLE with remedies.
- [ ] **Environment** shows either the CDS credentials as found, or the
      DATA UNAVAILABLE panel — not a traceback.
- [ ] **Run full analysis** completes and the progress bar clears.
- [ ] **Ledger** exports CSV, LaTeX and the HTML report; the stamp shows a real
      git commit hash, not `unavailable`.
- [ ] Change the seed, rerun, and confirm the stamp changes with it.
- [ ] Set the LJF model to RIGID and confirm C2 and C7 become
      `UNTESTABLE — DATA MISSING` rather than reporting zero.
- [ ] **Environment** shows a green MEASURED banner naming a NOAA station, and
      the constituent table lists current semi-major/semi-minor axes.
- [ ] Switch the sidebar **Tidal forcing** to the ASSUMED placeholder and confirm
      the banner and provenance chips change with it.

If the stamp reads `git commit: unavailable`, the container has no git metadata.
That is cosmetic on Streamlit Cloud; the CSV export still carries the version,
seed and geometry digest.

---

## 7. Tidal forcing

**Already real, out of the box.** Six NOAA CO-OPS stations are cached in
`data/constituents/` and committed, so the deployed app has MEASURED tidal
forcing — water-level amplitudes and tidal current ellipses — with no account,
no API key and no network call at runtime. The sidebar's **Tidal forcing**
selector picks between them.

To refresh them, or add stations:

```bash
PYTHONPATH=. python scripts/fetch_tides.py --force
```

These are real published constants but they are **not** the Arabian Gulf
platform site, and every claim that uses them says so. C3 returns the same
verdict at all six, so the site-specific extraction below changes the numbers
rather than the conclusion.

### Site-specific extraction (TPXO / FES)

Only worth doing locally — the model files are far too large for the container.

```bash
pip install -r requirements-data.txt
```

Download TPXO9-atlas from <https://www.tpxo.net/global> (free registration) or
FES2014 from AVISO, unpack it, and point the app at it:

```bash
export TIDETWIN_TIDE_MODEL_DIR=/path/to/tpxo9-atlas
```

Then extract and cache the constants for your platform position. Until an
extraction is verified against a known station, the app keeps reporting the
tidal constants as ASSUMED — which is the honest state, because an unverified
extraction relabelled as MEASURED would be worse than a placeholder that admits
what it is.

Meanwhile the sidebar accepts harmonic constants directly, typed in or uploaded
as CSV, for any site you like. Those are ASSUMED — nothing typed into a sidebar
can be MEASURED — and the report says so on every claim that uses them.

---

## Troubleshooting

**Build fails installing `scipy` or `pandas`.** Check `runtime.txt` says
`python-3.11` and that Advanced settings matched it. The pins in
`requirements.txt` are tested on 3.11.

**App exceeds the 1 GB limit.** You have almost certainly installed
`requirements-data.txt` on the cloud. `pyTMD` and `netCDF4` belong on a local
machine only.

**"Geometry SHA-256 does not match manifest".** A geometry table was edited
without regenerating the manifest. Either restore it with
`git checkout data/geometry/`, or, if the change was intentional, run
`python scripts/write_geometry_manifest.py` so the edit shows up in review.

**Figures export only as HTML.** `kaleido` did not install. The PNG and PDF
buttons disable themselves and say so; nothing else is affected.
