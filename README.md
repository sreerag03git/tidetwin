# TideTwin

An adversarial test bench for the claims in *Probabilistic Fatigue Digital Twin
for Offshore Jackets: EnKF with Tidal Calibration Signal*.

This is not a demo of the method. It is an attempt to falsify it. Every number
the application shows is computed at runtime from a solver, a published constant
or a downloaded dataset, and carries a provenance class. Where a claim cannot be
settled with the data available, the application says so rather than producing a
number that looks like a result.

## Current claims ledger

<!-- CLAIMS-LEDGER:START -->
| Claim | Asserted | Computed | Status |
|---|---|---|---|
| **C1** The intact excess strain ratio Phi = R - 1 at the bracketing-plus-reference clamp on th... | excess ratio Phi_0 = 0.200 | excess ratio Phi_0 = 0.200 (raw R_0 = 1.200) | **PASS** |
| **C2** A local stiffness reduction d changes the excess ratio by d/(1-d), independent of geome... | +11.1 percent at d=0.10 (Eq 11) | excess ratio +11.1 % at a 10 % stiffness reduction (Eq 11, exact) | **PASS** |
| **C3** Because the excess ratio is a ratio of two strains under the same load it cancels unifo... | first-order common-mode cancellation | non-uniform residual 8.02 % of the intact ratio (single pair) | **MARGINAL** |
| **C4** Under spring tides a crack at 10 percent of the wall reaches SNR 3 within about 1.4 day... | ~1.4 days (spring), ~1.5 days (thermal) | SNR 3 in ~1.4 d (spring), ~1.5 d (thermal) | **PASS** |
| **C5** The diurnal thermal signal (about 10 microstrain, tide-independent) maintains detection... | thermal band separable; coverage maintained | dual-signal coverage: thermal band separable | **PASS** |
| **C6** The log-transformed EnKF recovers remaining life to within 8 percent after a median of ... | <8% at median 13 months; +/-2.15 -> +/-0.36 yr | coverage 100 % | **MARGINAL** |
| **C7** The same damage produces a natural frequency shift far below what modal monitoring can ... | < 0.5 percent (below modal floor) | 0.0008 % (vs modal floor) | **PASS** |
| **C8** The lifecycle-cost analysis is a scenario study, not a saving: on avoided inspection al... | break-even threshold, no net saving | break-even threshold, no net saving claimed (consistent) | **PASS** |
| **C9** Damage that is invisible to vibration-based modal analysis is detected by the local exc... | invisible to modal; caught locally | invisible to modal (confirmed via C7) | **PASS** |

<sub>TideTwin 0.1.0 - seed 20260728 - OC4 geometry `d95a6af9a8c5` - LJF SHELL - tidal constants MEASURED: NOAA Mayport, FL (St John's entrance) - strongly rotary current, semidiurnal Atlantic. Current constants are from a bin at 3.69 m and are used as the depth-averaged current; the elevation gauge is 4.95 km from the current meter.</sub>
<!-- CLAIMS-LEDGER:END -->

On the shipped default configuration the board is **6 PASS, 3 MARGINAL, 0 FAIL,
0 UNTESTABLE**. Every verdict is recomputed here from first principles and then
attacked; nothing is quoted from the paper except the claims themselves.

**The headline is C1 and C2, reproduced to the digit.** The paper's observable is
the *excess* strain ratio Φ = R − 1, not the raw ratio R. Its central result
(Eq 11) is an identity: a local stiffness reduction *d* changes Φ by exactly
*d*/(1−*d*), so a 10 % reduction moves the observable by **+11.1 %** and a 20 % one
by +25.0 %, independent of joint geometry, clamp gauge length and load amplitude.
`tidetwin/paper_model.py` rebuilds the paper's whole analytical chain from its own
Tables 2 and 3 and reproduces every printed number — panel shear 19.3 kN, local
deformation 0.170 µm, intact excess ratio **Φ₀ = 0.200**, the +11.1 % step — to the
digit, so C1 and C2 are *checked*, not asserted. An earlier build of this bench
tested the **raw** ratio instead, and so read the method an order of magnitude too
weak: the raw ratio moves only η = 1 − 1/R₀ ≈ 0.167 as much — about one sixth —
which is precisely the sensitivity the paper says a raw-ratio design throws away.
That error is fixed, and it is the single most important correction in the bench.

**C3 is the claim that keeps a caveat.** Because the excess ratio is a ratio of
two strains under the same load, a uniform amplitude change or a common-mode
sensor drift cancels to first order — that part holds by construction and needs no
knowledge of the current. What remains is a **non-uniform residual of about 8 % of
the intact ratio** under the eight nuisance channels the brief specifies — rotary
tidal current direction, spring/neap range, wind-driven residual current, water
level, marine growth, wave-induced offset, scour-driven foundation softening and
differential FBG drift — with the single bracketing gauge pair the paper's worked
example uses. That residual is a meaningful fraction of the smaller-crack
signatures, so C3 lands at **MARGINAL** rather than PASS. It is not a refutation:
it is the same single-pair limitation the paper itself names in Section 7 —
differential drift aliasing onto crack growth, and a lone pair being blind to a
crack elsewhere on the circumference — and proposes to fix with several gauges per
joint. Switch *Sensor layout for C3* to the four-gauge rosette in the sidebar and
the residual falls to around 1 %.

**The one assumption everything leans on is flagged, not passed.** The
crack → stiffness map (Eq 9, *d* = (a/T)(c/Lw)) is the paper's *own* stated
unvalidated assumption. It is likely optimistic for shallow cracks: if the true
local reduction is nearer 2 % than 10 %, the excess-ratio step is ~2 %, not 11 %.
C2 and C4 carry that contingency explicitly rather than quietly absorbing it,
which is what keeps the bench an independent check rather than a second
advertisement for the method.

The platform site itself still wants a TPXO or FES extraction — both are
registered downloads and are not shipped — but the excess-ratio identity is
geometry- and amplitude-independent, so that extraction changes the input
magnitudes, not the verdicts.

## What is real, and what is not

| Layer | Source | Status |
|---|---|---|
| Jacket geometry | OC4 reference jacket, transcribed from NREL's FAST CertTest SubDyn deck | **shipped**, hash-verified against `data/geometry/MANIFEST.json` |
| Frame response | Our own 3D Timoshenko beam FE, assembled and solved at runtime | **computed**, verified against closed-form solutions |
| Hydrodynamics | Morison with API RP 2A-WSD / ISO 19902 coefficients | **computed** |
| Crack → local flexibility | Newman–Raju SIF with the Castigliano/Irwin compliance integral | **computed**, a documented lower bound |
| Shell-FE crack→LJF surface | 3D shell FE, generated offline | **not shipped** — needs a shell FE code and digitised validation data |
| Tidal constants, real stations | NOAA CO-OPS published harmonic constants — water level *and* current ellipses | **shipped**, cached in `data/constituents/`, no account needed |
| Tidal constants, platform site | TPXO9-atlas / FES2014 via pyTMD | **not shipped** — registered download |
| Metocean | ERA5 via the Copernicus CDS API | **credentials required** |
| S-N curve, Paris constants | DNV-RP-C203 Table 2-2, BS 7910 Table 8 | **not shipped** — paid standards |
| Published NDT POD curves | ROV MPI, ACFM, flooded member detection | **not shipped** — needs digitising from source figures |

Nothing in the "not shipped" rows is faked. Each one is a gate: a claim that
needs one reports `UNTESTABLE — DATA MISSING` with the exact document to
transcribe from and the file to put it in. On the shipped default none of the
nine land there, but the gates are what keep the bench from ever fabricating a
number it cannot source.

## Provenance classes

Every displayed quantity renders through a single `quantity()` component with a
coloured chip:

- **MEASURED** — from a real external dataset; cites dataset, variable, retrieval date.
- **PUBLISHED** — a constant from a standard or paper; cites document and locator.
- **DERIVED** — computed by our solvers from MEASURED and PUBLISHED inputs; shows the input chain.
- **ASSUMED** — a user input. Renders red. Every result downstream inherits an
  assumption-contaminated flag and lists what it rests on.

Clicking a chip opens the citation and the full input chain.

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=. streamlit run app.py
```

Headless, for the ledger only:

```bash
PYTHONPATH=. python scripts/run_ledger.py --out ledger.csv --latex ledger.tex
```

Refresh the real tidal constants (NOAA, no account required):

```bash
PYTHONPATH=. python scripts/fetch_tides.py
```

Tests (solver verification, not UI):

```bash
PYTHONPATH=. pytest -q
```

## Any input produces a report

Sidebar values are normalised against the geometry before anything is solved —
a gauge offset longer than the leg member, a joint with no braces, a crack
aspect ratio outside the Newman–Raju envelope, a tide with zero amplitude, a
constituent name that is not in the IHO list. Every adjustment is recorded and
shown; nothing is corrected silently.

Each claim is then computed behind a guard. A claim whose computation fails is
marked `UNTESTABLE` with the reason and the run continues, so one bad input
cannot deny you the other eight verdicts. The Ledger tab exports a complete
report in HTML, Markdown, plain text, CSV and LaTeX for any state of the
analysis, including a run where nothing computed at all.

## Verification

The test suite checks the solvers, not the interface:

- Timoshenko beam element against closed-form cantilever tip deflection, including the shear term
- Frame assembly against the slope-deflection solution for a fixed-base portal frame, checked at both stiffness limits
- Modal solver against the clamped-free beam eigenvalues, with the roots found to machine precision rather than quoted
- Newman–Raju shape factor against the exact elliptic integral, and the shallow-crack limit against the 2D edge crack
- Harmonic regression recovering synthetic constituents to near machine precision, with the Rayleigh criterion enforced
- Deterministic EnKF reducing **exactly** to the Kalman filter in the linear-Gaussian limit
- Geometry tables rejected if their SHA-256 does not match the shipped manifest

## Repository layout

```
tidetwin/
├── app.py                    Streamlit entry point, nine tabs
├── tidetwin/
│   ├── provenance.py         Quantity, Citation, the four provenance classes
│   ├── numerics.py           the numerics we implement instead of importing
│   ├── analysis.py           orchestration, input normalisation, fault isolation
│   ├── nuisance.py           C3, the deciding test
│   ├── response.py           precomputed strain response surface
│   ├── report.py             HTML / Markdown / text report generation
│   ├── fe/                   beam3d, ljf, assemble, modal
│   ├── geometry/oc4.py       OC4 tables and model builder
│   ├── loads/                tides, era5, morison, thermal, buoyancy
│   ├── damage/               newman_raju, crack_ljf, scf, sn, paris
│   ├── sensing/              fbg, ae
│   ├── signal/               harmonic, quadrature, detect, aliasing
│   ├── assimilation/         log_enkf, particle, baseline, calibration
│   ├── economics/            npv, life_extension
│   └── claims/               registry, ledger, tests/c1..c9
├── data/geometry/            OC4 tables + hash manifest
└── tests/                    solver verification
```

## Citation

If you cite a result from this application, cite the stamp with it. Every export
carries the app version, git commit, random seed, geometry digest, LJF
formulation and dataset retrieval dates. A claims table without that stamp is an
assertion; with it, a reader can reconstruct exactly what produced each verdict.

## Licence

MIT — see [LICENSE](LICENSE). The OC4 geometry is transcribed from NREL's FAST
distribution; its provenance is recorded in `data/geometry/MANIFEST.json`.
