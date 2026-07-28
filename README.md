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
| **C1** Under intact conditions the tidal strain ratio at a bracketing sensor pair on the targe... | 1.800 | 2.1974 (reciprocal 0.4551) | **FAIL** |
| **C2** A crack at the joint changes the strain ratio from 1.800 to 2.000, an 11.1 percent dama... | 11.1 percent (1.800 -> 2.000) | 0.0073 % at a/T=0.5, 2c=100 mm | **UNTESTABLE - DATA MISSING** |
| **C3** The tidal strain ratio is stable enough under environmental variation for the damage si... | nuisance sigma below one third of the damage signature | sigma = 13.74 % of the intact ratio | **FAIL** |
| **C4** Detection is achieved in 4 to 9 days at a signal-to-noise ratio of 3. | 4-9 days | never detected in 93 % of trials | **FAIL** |
| **C5** During neap tides a differential thermal channel of 5 to 15 microstrain provides a usab... | 5-15 microstrain | aliasing resolved; amplitude n/a | **UNTESTABLE - DATA MISSING** |
| **C6** The log-transformed EnKF converges on remaining life to within +/-0.9 years, outperform... | +/-0.9 years | coverage 100 % | **MARGINAL** |
| **C7** The same damage produces a natural frequency shift below 0.5 percent, so modal methods ... | < 0.5 percent | 0.0008 % | **PASS** |
| **C8** The monitoring system returns a net present value of 19.9 million USD. | 19.9 MUSD | 1.64 MUSD (all inputs ASSUMED) | **UNTESTABLE - DATA MISSING** |
| **C9** The tidal method offers a probability-of-detection advantage over ROV MPI, ACFM and flo... | favourable a90/95 | a90 not reached | **UNTESTABLE - DATA MISSING** |

<sub>TideTwin 0.1.0 - commit `273d1ce` - seed 20260728 - OC4 geometry `d95a6af9a8c5` - LJF SHELL - tidal constants ASSUMED placeholder - generated 2026-07-28T18:20:49Z</sub>
<!-- CLAIMS-LEDGER:END -->

**The headline finding is C3.** Under the eight nuisance channels the brief
specifies — rotary tidal current direction, spring/neap range, wind-driven
residual current, water level, twenty years of marine growth, wave-induced
offset, scour-driven foundation softening and differential FBG drift — the
standard deviation of the intact strain ratio comes out at **13.7 % of the
ratio**, against a claimed damage signature of 11.1 %. That is 1.24× the signal,
against a one-third limit.

Four things make that a decision rather than a number:

- **It is converged.** σ moves by 4.1 % across the second half of the run, inside
  the 5 % tolerance. An unconverged Monte Carlo has decided nothing, and the app
  withholds the verdict when it detects one.
- **It does not depend on our crack model.** The verdict is computed against the
  paper's *own* claimed 11.1 % signature as well as against the one this app
  computes, and fails both. The first comparison involves no modelling choice
  made here.
- **It is not an artefact of the assumed range widths.** Every nuisance range
  would have to shrink to **0.31× its assumed width, simultaneously on all eight
  channels**, for C3 to pass — a sea roughly three times quieter than assumed, on
  every axis at once.
- **The channels partially cancel, and it still fails.** The joint variance comes
  out 73 % *below* the sum of the individual ones: the ratio normalisation
  genuinely does reject a large part of what each channel does alone. That is a
  point in the method's favour, and the margin is still not there.

The storm-driven channels — wind current, wave offset and surge — are drawn
correlated rather than independently, since they share a cause. Drawing them
independently would have understated the joint σ and flattered the method.

## What is real, and what is not

| Layer | Source | Status |
|---|---|---|
| Jacket geometry | OC4 reference jacket, transcribed from NREL's FAST CertTest SubDyn deck | **shipped**, hash-verified against `data/geometry/MANIFEST.json` |
| Frame response | Our own 3D Timoshenko beam FE, assembled and solved at runtime | **computed**, verified against closed-form solutions |
| Hydrodynamics | Morison with API RP 2A-WSD / ISO 19902 coefficients | **computed** |
| Crack → local flexibility | Newman–Raju SIF with the Castigliano/Irwin compliance integral | **computed**, a documented lower bound |
| Shell-FE crack→LJF surface | 3D shell FE, generated offline | **not shipped** — needs a shell FE code and digitised validation data |
| Tidal constants | TPXO9-atlas / FES2014 via pyTMD | **not shipped** — multi-gigabyte registered download |
| Metocean | ERA5 via the Copernicus CDS API | **credentials required** |
| S-N curve, Paris constants | DNV-RP-C203 Table 2-2, BS 7910 Table 8 | **not shipped** — paid standards |
| Published NDT POD curves | ROV MPI, ACFM, flooded member detection | **not shipped** — needs digitising from source figures |

Nothing in the "not shipped" rows is faked. Each one is a gate: the dependent
claim reports `UNTESTABLE — DATA MISSING` with the exact document to transcribe
from and the file to put it in. Several claims land there, and that is the
correct answer rather than a shortfall.

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
