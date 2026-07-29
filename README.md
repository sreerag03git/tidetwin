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
| **C1** Under intact conditions the tidal strain ratio at a bracketing sensor pair on the targe... | 1.800 | 1.9059 (reciprocal 0.5247) | **MARGINAL** |
| **C2** A crack at the joint changes the strain ratio from 1.800 to 2.000, an 11.1 percent dama... | 11.1 percent (1.800 -> 2.000) | 0.0043 % at a/T=0.5, 2c=100 mm | **UNTESTABLE - DATA MISSING** |
| **C3** The tidal strain ratio is stable enough under environmental variation for the damage si... | nuisance sigma below one third of the damage signature | sigma = 8.19 % of the intact ratio | **FAIL** |
| **C4** Detection is achieved in 4 to 9 days at a signal-to-noise ratio of 3. | 4-9 days | never detected in 88 % of trials | **FAIL** |
| **C5** During neap tides a differential thermal channel of 5 to 15 microstrain provides a usab... | 5-15 microstrain | aliasing resolved; amplitude n/a | **UNTESTABLE - DATA MISSING** |
| **C6** The log-transformed EnKF converges on remaining life to within +/-0.9 years, outperform... | +/-0.9 years | coverage 100 % | **MARGINAL** |
| **C7** The same damage produces a natural frequency shift below 0.5 percent, so modal methods ... | < 0.5 percent | 0.0008 % | **PASS** |
| **C8** The monitoring system returns a net present value of 19.9 million USD. | 19.9 MUSD | 1.64 MUSD (all inputs ASSUMED) | **UNTESTABLE - DATA MISSING** |
| **C9** The tidal method offers a probability-of-detection advantage over ROV MPI, ACFM and flo... | favourable a90/95 | a90 not reached | **UNTESTABLE - DATA MISSING** |

<sub>TideTwin 0.1.0 - commit `9f07b32` - seed 20260728 - OC4 geometry `d95a6af9a8c5` - LJF SHELL - tidal constants MEASURED: NOAA Mayport, FL (St John's entrance) - strongly rotary current, semidiurnal Atlantic. Current constants are from a bin at 3.69 m and are used as the depth-averaged current; the elevation gauge is 4.95 km from the current meter. - generated 2026-07-29T14:18:36Z</sub>
<!-- CLAIMS-LEDGER:END -->

**The headline finding is C3.** Under the eight nuisance channels the brief
specifies — rotary tidal current direction, spring/neap range, wind-driven
residual current, water level, twenty years of marine growth, wave-induced
offset, scour-driven foundation softening and differential FBG drift — and under
**real measured tidal forcing** at the shipped default station, the standard
deviation of the intact strain ratio comes out at **8.2 % of the ratio**, against
a claimed damage signature of 11.1 %. That is 0.74× the signal, against a
one-third limit — so it misses by a factor of 2.2.

That default is the *most favourable* of the seven regimes tested. The others run
to 39.6 %.

Four things make it a decision rather than a number:

- **It is converged.** σ moves by 2.5 % across the second half of the run, inside
  the 5 % tolerance. An unconverged Monte Carlo has decided nothing, and the app
  withholds the verdict when it detects one — this station needed 900 samples,
  and at 250 the verdict was correctly withheld.
- **It does not depend on our crack model.** The verdict is computed against the
  paper's *own* claimed 11.1 % signature as well as against the one this app
  computes, and fails both. The first comparison involves no modelling choice
  made here.
- **It is not an artefact of the assumed range widths.** Every nuisance range
  would have to shrink to **0.45× its assumed width, simultaneously on all eight
  channels**, for C3 to pass — a sea more than twice as quiet as assumed, on every
  axis at once.
- **The channels partially cancel, and it still fails.** The joint variance comes
  out 33 % *below* the sum of the individual ones: the ratio normalisation
  genuinely does reject a large part of what each channel does alone. That is a
  point in the method's favour, and the margin is still not there.

The storm-driven channels — wind current, wave offset and surge — are drawn
correlated rather than independently, since they share a cause. Drawing them
independently would have understated the joint σ and flattered the method.

### C3 does not depend on the tide

The obvious remaining objection was that the FAIL might be an artefact of the
placeholder tide. It is not. `scripts/settle_c3.py` reruns the whole budget
against **real published harmonic constants** from six NOAA CO-OPS stations —
water-level amplitudes *and* tidal current ellipses — spanning semidiurnal to
mixed-diurnal regimes, rectilinear to strongly rotary currents, and nearly an
order of magnitude in current amplitude:

| Tide | M2 current | Ellipse ecc. | Form factor | Dispersion, % of ratio | ÷ claimed | Break-even | Verdict |
|---|---|---|---|---|---|---|---|
| Placeholder *(not real)* | 0.250 | 0.320 | 0.571 | 13.7 | 1.23 | 0.32× | FAIL |
| Mayport, FL | 0.584 | 0.299 | 0.184 | 8.0 | 0.72 | 0.47× | FAIL |
| Woods Hole, MA † | 0.496 | 0.051 | 0.458 | 15.0 | 1.35 | 0.24× | FAIL |
| Friday Harbor, WA | 0.381 | 0.178 | 1.733 | 17.4 | 1.57 | 0.22× | FAIL |
| Boston, MA | 0.100 | 0.098 | 0.164 | 27.3 | 2.46 | 0.18× | FAIL |
| Kings Bay, GA | 0.564 | 0.078 | 0.174 | 27.5 | 2.47 | 0.13× | FAIL |
| Richmond, CA | 0.526 | 0.055 | 0.792 | 39.6 | 3.57 | never | FAIL |

† variance undefined; robust scale used. Every other row is a standard deviation.

Every real regime fails, by between 2.2× and 11× the limit. The placeholder was
if anything *generous* — it sits in the lower half of the real range rather than
stacking the deck. Reproduce with
`python scripts/settle_c3.py --samples 700`.

**The mechanism, which is the useful part.** Nuisance dispersion tracks the
*shape* of the tidal ellipse (r = **−0.70** against eccentricity) and is
essentially independent of its *size* (r = −0.17 against current amplitude). A
rotary current never goes slack, so the strain ratio stays well conditioned; a
reversing current passes through zero twice a cycle and the ratio degrades near
slack water. The method is not signal-starved — it is ill-conditioned.

**At a reversing-current site the statistic has no variance at all.** The ratio's
denominator is the upper gauge's M2 amplitude, which approaches zero at slack
water, and a ratio with a near-zero denominator is Cauchy-like. The sample
standard deviation then *grows* with sample count instead of converging — it went
38.7% → 53.4% between 700 and 2800 samples at Woods Hole and was still climbing.

The app detects this from the fact that a **robust scale converges while σ does
not**: a robust scale converges for any distribution with a density, the standard
deviation only if the variance exists. Where the variance does not exist, C3 is
decided on 0.7413 × IQR rather than withheld, because more samples cannot fix a
statistic that has no variance. Woods Hole still fails, at 15.0% against the
claimed 11.1%.

So eccentricity predicts two things, and the second matters more than the first:
the *level* of the nuisance floor, and whether the statistic is well posed at all.
The two most reversing sites are where σ/robust is largest (2.8 at Woods Hole
against 1.0–1.3 elsewhere), which is where the variance breaks down.

That a detection statistic has no second moment at an ordinary offshore site is a
finding against the method in its own right: a detection threshold cannot be set
from a quantity that has no variance.

Actionable version: if this method is deployed anywhere, it wants a strongly
rotary tidal current. And at the most rotary site tested it still misses by 2.2×.

The platform site itself still needs a TPXO or FES extraction — both are
registered downloads and are not shipped. But the evidence says that extraction
would change the numbers, not the conclusion.

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

Refresh the real tidal constants (NOAA, no account required):

```bash
PYTHONPATH=. python scripts/fetch_tides.py
```

Re-run the C3 settlement across every real tidal regime:

```bash
PYTHONPATH=. python scripts/settle_c3.py --samples 300
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
