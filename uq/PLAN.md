# Phase 2 — Uncertainty module + surrogate (advisor-mode plan)

Scope decision (2026-09-19): **robust-only paper** (deterministic vs robust
comparison under uncertainty). RBDO = paper #2. Work mode: Jaminur implements,
Claude specifies & verifies. Companion to `validation/` (Phase 1, closed).

Measured basis (2026-09-19): one `evaluate_shape` call ≈ **13 s** (3 mid-design
points: 13.1/12.1/14.6 s). DOE of 500 pts ≈ 1.8 h — run overnight, rerun freely.

---

## Uncertainty model (v1 — keep it to 4 vars)

| Var | Symbol | Distribution | Rationale (cite in paper) |
|---|---|---|---|
| Entry flight-path angle | gamma0 | N(−2°, 0.5°) 1σ | Typical EI FPA dispersion, entry-corridor studies |
| Entry velocity | V0 | N(7830, 50) m/s | ~0.6% navigation/deorbit budget |
| Drag model-form scale | k_CD | N(1.00, 0.05) | V2 validation: M=10 CD 1.610 vs WT 1.523 = +5.7%; M=3 shape 6.1% |
| Lift model-form scale | k_CL | N(1.00, 0.05) | V2 absolute L/D error ≤ 0.029 ≈ 10% of L/D ≈ 0.3 → 5% per coefficient |

Deferred (do not build now): mass dispersion, atmospheric density family,
correlated k_CD/k_CL. Each is a one-line addition to the table if week 1 is fast.

Surrogate input space: **x = (rn, rs, r_theta, gamma0_deg, V0_km_s, k_CD, k_CL)** — 7 dims.
Outputs (one GP per scalar): **Qs, sg, q_stag_pk, n_tot_pk** (eta_V is pure
geometry = f(design vars) — compute exactly, never surrogate it).

## Step 1 — `capsule_opt/uncertainty/` package

New package, two files:

- `__init__.py` — empty.
- `variables.py`:

```python
from scipy import stats

# name, scipy frozen distribution, hook
UNCERTAIN_VARS = [
    ("gamma0_deg", stats.norm(-2.0, 0.5)),      # entry FPA [deg]
    ("V0",         stats.norm(7830.0, 50.0)),   # entry speed [m/s]
    ("k_CD",       stats.norm(1.00, 0.05)),     # drag model-form scale
    ("k_CL",       stats.norm(1.00, 0.05)),     # lift model-form scale
]

def sample(n, seed=42):
    """(n, 4) array of samples via independent inverse-CDF of a unit LHS."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(UNCERTAIN_VARS), seed=seed).random(n)
    import numpy as np
    return np.column_stack([dist.ppf(u[:, j]) for j, (_, dist) in enumerate(UNCERTAIN_VARS)])
```

Self-test (`python -m capsule_opt.uncertainty.variables`): sample(10000) →
means within 2% of nominal, stds within 3%, min/max sane (no |gamma0| > 5σ).

## Step 2 — hook aero scales into `evaluate_shape`

`capsule_opt/optimization/objectives.py`: add kwarg `aero_scales=(k_CD, k_CL)=None`
and wrap right after `build_aero_database`:

```python
if aero_scales is not None:
    k_CD, k_CL = aero_scales
    def _s(interp, k):
        def w(x): return k * interp(x)
        return w
    interp_CD, interp_CL = _s(interp_CD, k_CD), _s(interp_CL, k_CL)
```

Call sites are `interp_CD(np.array([[M, alpha]]))[0]` (trajectory.py:62,
metrics.py:131) — the closure matches that signature as-is. Note: alpha_trim
comes from CM (unscaled) → trim is unaffected by k_CD/k_CL, as intended.
`entry_conditions` kwarg already exists → gamma0/V0 need no code.

**Regression (must pass before anything else):** `aero_scales=(1.0, 1.0)` at
(0.5,0.5,0.5), (0.4,0.3,0.6), (0.6,0.4,0.4) reproduces today's objectives to
float equality: eta_V 0.8347657, Qs 162660012.1, sg 8386376.6 (point 1).

## Step 3 — DOE runner `uq/doe.py` (repo-root run: `python -m uq.doe`)

- Inputs: `N=500` LHS over the 7-dim space: design vars uniform [0,1]³ (own
  qmc LHS, seed 1), uncertain vars from `variables.sample(N, seed=2)`.
- Loop `evaluate_shape(rn, rs, r_theta, aero_scales=(k_CD, k_CL),
  entry_conditions={'gamma0_deg': g, 'Vo': V})`; record the 4 outputs +
  `feasible` + all constraint values (for later robust constraints).
- **Checkpoint/resume: write `uq/doe_results.csv` after every point**
  (append + flush); skip already-done rows on restart. A 2 h run must survive
  a crash. Print progress every 10 pts with ETA.
- Infeasible/non-converged points: record NaN outputs, keep the row, count them
  (report at end; >5% = tell Claude before proceeding).

## Step 4 — surrogate `uq/build_surrogate.py`

- sklearn `GaussianProcessRegressor`, one per output. Normalize inputs
  (gamma0 in deg, V0 in km/s to keep O(1); or a StandardScaler) and outputs.
  Kernel: ConstantKernel * Matern(nu=2.5, length_scale=[...], ARD) + WhiteKernel.
  `n_restarts_optimizer=10`, `random_state=0`.
- Split 450 train / 50 test (fixed seed). Report per output: RMSE_test,
  RMSE / range, R²_test (target: **R² > 0.95 and RMSE < 2% of range** for Qs,
  sg; < 3% for peaks). Save models + scaler to `uq/gp_<output>.joblib`.
- Print a Q²-style table. This is the **go/no-go checkpoint**: if any output
  misses target, do NOT tune blindly — bring the table to Claude (options:
  more DOE pts, log-transform Qs, separate GP per regime).

## Step 5 — sanity checks (before declaring week 1 done)

- Deterministic corner check: surrogate at nominal z (gamma0=−2, V0=7830,
  k=1) vs direct `evaluate_shape` at ~10 random design points → within the
  Step-4 error bands.
- Objective ranges vs thesis Table 6.2 (sg 2930–4695 km was the thesis Pareto
  envelope at gamma=−2°): our full design-space DOE will exceed it (probe point
  (0.5,0.5,0.5) gave sg ≈ 8386 km, feasible) — confirm the *envelope* contains
  the thesis range and note where it doesn't.

## Step 6 — DECIDED 2026-10-08: Option A (statistical moments)

Option B (chance constraints) rejected: reserved territory of paper #2 and
the 2025 RESS capsule RBMDO paper, and its optima sit in the GP tails.

- Objectives (4): max eta_V, min E[Qs], max E[sg], min std[Qs].
- Constraints: pitch-stability + trim-on-nose exact (z-free geometry);
  peaks as E[q] + 2·sigma_q <= limit (config: 700 kW/m², 1000 kW/m², 5 g);
  P(slow) <= 0.05 via the family classifier — carries the range-dispersion
  story, since std[sg] is not pointwise-learnable (sg-slow irreducible).
- Machinery: det and robust fronts BOTH on the surrogate (det = surrogate at
  nominal z0 = (-2°, 7830, 1, 1)); one fixed 2000-pt z-LHS (seed 42) reused
  for every candidate (common random numbers → smooth fitness); post-hoc
  fragility audit of the det front through the same engine; verification MC
  at selected designs (simulator, overnight, resumable).
- Probe-0 FIRST (`uq/probe_dispersion.py`, 5 designs × 60 sims, t_max=14400,
  stages to uq/data/probe_dispersion.csv, resume-safe): true sigma_z at fixed
  geometry vs final-gate RMSE. Verdict rule: sigma/RMSE > ~2 → plain std[Qs]
  objective; ~1 → inflate with GP predictive variance. Smoke (2 rows, D0):
  Qs 3.10, q_stag 2.43, q_shldr 1.94, n_max 0.26 — n_max dispersion may be
  BELOW surrogate error → its E+2sigma constraint would be error-dominated;
  decide after the full probe.

### Step-6 machinery + pre-flight findings (2026-10-08/09, built while probe ran)

- `uq/engine.py` — MomentEngine: CRN z-LHS (fixed seed) → per-candidate
  E/sd (plain + GP-variance-inflated via law of total variance), P(peak >
  limit), P(slow), P(capture); det point predictions at nominal z0; eta_V
  exact. Geometry constraints surrogated (exact = aero-db build per
  candidate): trim-existence classifier 98.5% OOF (in the DOE pitch-stable
  <=> trim exists — 473/2800 no-trim sentinels are exactly the unstable
  rows) + within-trim nose HGBR 98.8% sign-acc → artifacts/geom_margins.
  Self-test (`python -m uq.engine`): DOE-row point predictions 0–8% vs CSV
  (gate-consistent); mixture sg residuals larger on slow rows = known
  irreducible branch.
- `uq/optimize.py` — det/robust pymoo problems (batched), robust = 4 obj +
  E+2·sd ≤ limit peaks (sd infl|plain) + geometry [+ P_slow ≤ budget];
  two-tier CRN: n_z=500 in-loop (~0.65 s/cand under load), final front
  re-evaluated at n_z=2000. Usage in module docstring; smoke-passed both
  modes; full robust run ≈ 4–6 h at pop 150 × 200 gens.
- `uq/diagnostics/diag_robust_feasible.py` (300-design LHS scan, n_z=500):
  peak constraints NON-binding in the interior (98–100% feasible even at
  E+2σ_infl — they bind only in the sharp corners the det front will hug);
  geometry binds (~51%); **P_slow = 0.34–0.59 over the WHOLE design space →
  any small budget is infeasible (0/300 at 0.05)** — loft probability is set
  by the γ0 dispersion, not by geometry; corr(P_slow, E_sg) = +0.72,
  corr(P_slow, σ_Qs_infl) = +0.74 (loft buys range with heat-load
  dispersion). **USER DECISION 2026-10-09: P_slow is REPORT-ONLY** (no
  constraint; carried as a coloring variable + paper finding). CLI default
  'off' reflects this.

### Both fronts done + audit (2026-10-09)

Robust run: 10.5 h overnight, pop 150 × 200 gens, 100% feasible →
`uq/data/opt_robust.csv` (n_z=2000 report columns included). Det audit
(`uq/audit.py`, 426 s): σ_Qs median 30.3 MJ/m² (CV 22.8%), P_exc(q_stag)
>1% on 25% of det designs (max 6.8%), P_exc(q_shldr) max 10%, P(n_max)
≈ 0, robust-feasible 92% of det front. **Headline**: matched-performance
pairs (E_Qs +2%, sg −2%): 5371 pairs, 98.1% favor robust on σ_Qs; median
σ cut 35.0% best-match-per-design (fig2 ECDF median; pair-level median
27.5%). [CORRECTED 2026-10-10: the previously quoted 43.5% was a one-off
audit-day mis-definition — not reproducible under any standard window;
authoritative numbers in paper/tables/table4_matched_pairs.csv.]
Robust front worst-member P_exc halves (4.8%/6.6% vs
6.8%/10%); σ floor identical on both fronts — **16.4 MJ/m² / CV 19.9% =
irreducible dispersion floor set by the uncertainty model, not shape**.
Strict 4-way dominance rare (30/30000) as expected. Next: figures.py +
verification MC at selected designs.

### Det baseline done (2026-10-09)

`python -m uq.optimize det 200 300` — 203 s, seed 1, final pop 200 all
feasible → `uq/data/opt_det.csv` (designs + F/G + point predictions incl.
P_slow for the post-hoc fragility audit). Remaining: probe verdict (σ form
for sd_Qs objective / n_max constraint) → robust run (user-launched,
~4–6 h at pop 150 × 200 gens) → det-front audit → figures.

### Knee designs + strict fronts + det sg-inflation finding (2026-10-09, `uq/knee.py`)

Both final pops already strictly non-dominated at report precision (det
200/200 on (−η_V, Qs, −sg); robust 150/150 on (−η_V, E[Qs], −E[sg], σ_Qs,infl)
at n_z=2000, not the loop's n_z=500 F). Knees: det = row 124 (anchor-hyperplane
knee); robust = row 51 (**pseudo-weights** — anchors degenerate: one design
anchors both E[Qs] and σ_Qs; pw_dev 0.0059). Third column: robust design
nearest the det knee in the (E[Qs], E[sg]) plane (fig2's ±2% window misses the
knee by 0.05% on E[sg] — brittle threshold, nearest is parameter-free).
Table (simulator at z0 + robust lens n_z=2000): `uq/data/knee_designs.csv` +
`knee_sim_z0.csv` cache. Sim at z0: det knee (Rn 5.60 m, θc 37.0°, η_V 0.745)
Qs 128.8 MJ/m², sg 7,465 km; robust knee (Rn 4.84 m, θc 21.6°, η_V 0.857) Qs
183.9, sg 11,321 km — **the robust knee dominates the det knee nominally on
range** despite its higher E[Qs]; matched robust design: E[Qs] −5.8%,
E[sg] −2.1%, σ cut only **3.7%** — the det knee itself sits near the σ floor,
so the robust advantage concentrates in the higher-performance region (fig2).

**FINDING (paper material): det-front sg(z0) inflated by slow-GP
extrapolation.** 54/200 det designs have surrogate sg(z0) > 20,015 km
(physical great-circle max); at the det knee the slow-family GP extrapolates
(μ_s → m_s = 101,226 km, log-sd 0.39 at z0 / up to 1.2 over the LHS) giving
sg(z0) = 26,004 km vs simulator 7,465 km = **3.5× overprediction**; NSGA-II
maximized sg straight into this epistemic artifact. Qs unaffected (1.7% at
z0). Robust front clean: E_sg spans 4,117–6,393 km, no inflated designs.
All headline comparisons (audit, fig2) already use the same CRN robust lens
on both fronts — fair; disclose in the surrogate-limitations paragraph as
"point-prediction optimization exploits epistemic extrapolation; the moment
formulation self-regulates". Verification MC should include the det knee.

### `uq/verify_mc.py` written + smoke-passed (2026-10-09; run = USER)

5 designs x 300 true-simulator z-samples (z-LHS fixed at 500, seed 2026,
sliced; t_max=14400): 0 det_knee, 1 rob_knee, 2 rob_matched (knee table) +
3 det_fragile (det front max audit Pexc q_shldr — the audit headline) +
4 rob_floor (robust front min sd_Qs_infl — the 16.4 MJ/m² floor claim).
Resume-safe staging to uq/data/verify_mc.csv; summary compares MC vs
MomentEngine n_z=2000 CRN: E/sd[Qs], E[sg], P_slow, P_cap, peak E+2σ and
exceedance counts (rule of three). Smoke (2 sims) passed. Launch (from
capsule_opt, foreground+Tee, ~10–12 h):
`C:\Users\xamii\anaconda3\python.exe -m uq.verify_mc | Tee-Object -FilePath uq\logs\verify_mc_log.txt`
(300 3 = knee-only ~6–7 h; summary auto-prints at end; re-run to re-print).

### VERIFY-MC DONE (2026-10-10, 9.7 h): Qs / sigma / P_exc VERIFIED; E[sg] biased high

5×300 sims, 0 failures, 1,499/1,500 captured (sur P_cap ~0.99), ~23 s/sim.
Per design vs surrogate (n_z=2000 CRN):

| check | result |
|---|---|
| E[Qs] | −3.2% … +1.4% (≤2.6 SE) across all 5 — **VERIFIED** |
| σ_Qs (MC vs infl) | MC 15.4–65.9 vs infl 16.4–66.7 — **VERIFIED** (floor design MC 15.4 vs claimed 16.4: floor real, slightly conservative) |
| peak E+2σ | within ~1–3% everywhere — **VERIFIED** |
| **P_exc at det_fragile** | q_shldr **29/299 = 9.7% vs sur 10.0%**; q_stag 7.4% vs 6.8%; n_max 0.3% vs 0.2% — **audit headline CONFIRMED by true simulator** |
| P_slow | MC 34.7–48.8% vs sur 39.1–49.0% (within ~6 pp) — OK |
| E[sg] | **OVERPREDICTED on all 5 designs: +8.8…+28.1% (2.7–10.1 SE)** — mixture lognormal-mean inflation (same mechanism as the det-knee sg(z0) finding), now measured under the CRN |

Paper handling: absolute E[sg] carries ~10–30% surrogate bias — disclose +
report verified-true MC E[sg] for the 5 designs; matched-pair comparison
uses E[sg] on BOTH fronts with uniform bias direction → ranking robust,
absolute values conservative-high. The load-bearing claims (σ_Qs objective,
E+2σ constraints, audit P_exc) are simulator-verified. Files:
`uq/data/verify_mc.csv`, `verify_mc_summary.csv`, `uq/logs/verify_mc_log.txt`.

### Paper area restructured (2026-10-10): `paper/` + `uq/tables.py`

figures.py now writes to `paper/figures/` (three files renamed:
fig2_matched_sigma_cut, fig5_design_space_sigma_map, fig6_probe_true_sigma;
old uq/figures/ deleted). Frozen descriptive-named data copies in
`paper/data/` (9 files), manuscript tables via NEW `uq/tables.py` →
`paper/tables/` (table1 knee designs physical units, table2 verification
copy, table3 audit headline, table4 matched pairs), interactive HTML in
`paper/supplementary/`. Provenance map: `paper/README.md` (uq scripts →
paper assets, never edit paper/ copies directly). Pipeline untouched
(knee.py/audit.py/verify_mc.py still read/write uq/data).

### Reviewer hardening trio DONE (2026-10-10)

1. **sanity_check.py FIXED for v2** (engine-based, t_max=14400, capture-skip,
   branch-matched sg): 10 random designs at z0 — Qs/q_stag/q_shldr/n_max all
   PASS (medians 0.35–1.34% vs gate bands); sg-slow PASS (5.1% vs 15%
   irreducible); sg-fast median 3.8% == band with one 12% outlier in 9 pts
   (sampling noise around the n=2800 gate; not a contradiction); mixture
   mean vs single realization documented as structural branch gap (E[sg]
   bias is verified by verify_mc instead). Envelope: feasible-DOE sg
   [1,524, 19,090] km contains thesis Table 6.2 ✓.
2. **Det seed repeatability** (optimize.py gained SEED argv; seeds 2/3 =
   189/186 s, 100% feasible, opt_det_s{2,3}.csv): objective stats
   near-identical across seeds (η_V medians 0.805–0.811; sg medians
   15.4–15.7 Mm); merged ND set 508/600 with contributions 166/167/175;
   pairwise cross-domination ≤ 14%. `uq/seed_check.py` → paper/tables/
   table5_det_seed_repeatability.csv.
3. **Convergence evidence** (`uq/convergence.py` → paper/supplementary/
   figS1_nsga2_convergence.{png,pdf}): fronts fill by gen 3–4 in all runs;
   ε decays to 6e-3–1e-2 recombination floor and stays there.

New files this session (uncommitted): sanity_check.py (rewritten),
optimize.py (seed arg), convergence.py, seed_check.py, figS1, table5,
opt_det_s2/s3.csv + logs.

### Manuscript skeleton + paper organization (2026-10-10, end of day)

ALL COMPUTE IS DONE (see blocks above). Since then: repo organized into
`paper/{figures,data,tables,supplementary,thesis_assets,manuscript}`
(provenance map paper/README.md; uq/ pipeline untouched; references/ +
validation_data/ moved OUTSIDE the repo by user -- build_reference_data.py
parents[2] still resolves); 11 __pycache__ purged; thesis folder inventoried
(THESIS_ASSETS.md: silhouettes/validation/restyle templates + docx media
extracted). Manuscript: main.tex (elsarticle) + sections/01-08 via \input,
References LAST (floats [htbp] + \clearpage fix), refs.bib 17 entries (9
[VERIFY] markers to fill), draft abstract, 5 data tables, 6 live figures.
Compiles 11 pp / 0 errors. NEXT: draft prose sec 01 (intro +
differentiation) first; author details pending from user.

## Explicitly NOT in this cycle

RBDO/probabilistic constraints (paper #2) · density/mass uncertainty families ·
PCK (plain GP only unless it fails targets) · V2-prime D-3890 validation
(deferred) · shape–guidance co-design (paper #3 candidate).

---

## Progress log

**2026-09-20 — Step 4 executed; gate FAILED for Qs/sg/n_max. Step 5 blocked.**

`uq/build_surrogate.py` + `uq/sanity_check.py` written (sanity NOT run — it
needs a passing surrogate). Both fitting spaces tried (raw | log-y; metrics
always computed in original units; 450/50 split, seed 0, Matérn 5/2 ARD +
WhiteKernel, 10 restarts):

| output | raw R² / RMSE%rng | log R² / RMSE%rng | gate 0.95 / 2%(3%) |
|---|---|---|---|
| Qs | 0.818 / 7.40 | 0.860 / 6.49 | FAIL |
| sg | 0.669 / 7.97 | 0.787 / 6.40 | FAIL |
| q_stag_max | 0.952 / 2.82 | 0.960 / 2.58 | **PASS (log)** |
| q_shldr_max | 0.939 / 3.16 | 0.949 / 2.90 | fail by 0.0013 on R² |
| n_max | 0.873 / 6.37 | 0.847 / 6.99 | FAIL (raw better) |

Diagnosis (all cheap tests done, none sufficient): log-transform buys ~1 pt;
poly2 trend in log space weak (CV-R²: sg 0.16, Qs 0.57) — no universal-Kriging
shortcut; no clean regime split (sg median smooth in γ₀; the 7% long-range
pts >10,000 km are scattered γ₀×V₀×geometry combos, mostly infeasible). GP
assigns 7–20% WhiteKernel noise to a noiseless simulator → response wiggles
below 500-pt resolution. Failures are the integral quantities (Qs, sg, n_max);
local peaks pass. **Verdict: data-limited → extend the DOE** (the plan's
remaining sanctioned option).

DECISION PENDING (user stopped here for the day): extend to 1000 total
(+500 pts ≈ 2.7 h) or 1500 (+1000 ≈ 5.3 h, overnight). Command:
`python -m uq.doe 1500` — resumes at row 500 and appends (the changed-N LHS
design yields fresh distinct points; union coverage is fine for GP fitting).

Next `build_surrogate` run on the extended DOE: switch the single 50-pt test
to 5-fold CV (mean ± std R²/RMSE) for a sturdier gate. On-disk `gp_*.joblib`
are the log-space fits from today — diagnostic only, everything refits on the
bigger DOE. Useful for the paper later: peak-heating ARD says k_CD/k_CL nearly
irrelevant (ls ≈ 32–47) — peak q̇ is set by local ρ,V at peak deceleration.

## 2026-10-12 — REVISION LOOP (external review of the full draft)

External review verdict: publishable core, but four hardening asks before
submission. Response plan (Phase A zero-compute + Phase B overnight + Phase C
user inputs):

- **A1/A2 `uq/sens.py`** — re-evaluate BOTH saved fronts through the moment
  engine under alternative uncertainty models (sigma_k 0.03/0.07, rho(k)=0.8,
  sigma_gamma0 0.25/0.75 deg; same LHS seed 42, gamma0/V0 quantiles shared
  with base) + matching-window sensitivity (1/2/3/5%). Output:
  paper/tables/table6_sensitivity.csv. 'base' variant must reproduce the
  published numbers (5,371 / 98.1% / 27.5% / 35.0%; floor 16.4; Pexc shldr
  max 10.0%) — correctness anchor.
- **A3 Appendix A** (sections/09_appendix_validation.tex) — identity-violation
  table, guidance variant family, deviations list: the reference-case
  forensics moved INTO the paper (reviewer: "a reviewer will need to see the
  evidence").
- **A4 manuscript edits** — "component-validated" phrasing (not "validated
  1–3% where it matters"), numerics-vs-mission separation sentence,
  claims-dependency paragraph (end of Sec 2.4), k-independence
  acknowledgment (Sec 3.1), E+2sigma non-interpretation sentence (Sec 4.2),
  bias/window cross-refs (Sec 7), Sec 5.7 sensitivity subsection w/ table.
- **B (user-launched nights)** robust seeds 2+3 (`python -m uq.optimize
  robust 150 200 infl off 2` then `3`) → seed_check 3+3 table + per-seed
  headline stats; **`uq/verify_mc_guidance.py`** (written+patch-checked
  10-12: Eq-2.40-as-printed law via make_gated_bank, 5 designs x 300 to
  uq/data/verify_mc_g40.csv, MC-vs-MC summary) — "conclusions survive the
  guidance law" evidence.
- **C** RESS-2025 + Hashemi refs from user; supervisor pass.

NOT doing: front re-optimization under the alternate law (within-CRN argument
+ MC guidance bracket instead), 6-DOF, density dispersion (paper #2).

## 2026-10-14 — THIRD external review answered (matching + range-bias audit)

Reviewer verdict: 2 critical / 3 high / 1 medium. Code audit confirmed
both criticals are real: the 35.0% design-level median is
`groupby('i')['cut'].max()` (outcome-selective), and the sec-6 "bias
acts in the same direction on both fronts so matching is unaffected"
argument was invalid — our own MC measures the differential.

New work, all zero-simulator-compute (front CSVs only):

- **`uq/matching.py`** — matching-construction audit; base reproduces
  5,371 / 98.1% / 27.5% / 35.0% exactly (anchor).
  * Prespecified NEAREST-eligible-match rule (min relative distance in
    (E_Qs, E_sg); ties by front index): **84.2% of designs beaten,
    median cut 4.3%**. Gap vs the envelope is structural: eligible
    partners span a median σ-factor 1.49 at matched performance (σ is
    traded against η_V); nearest partners sit at the 11th σ-percentile
    of the robust front in the low-E_Qs tercile (both fronts on the
    floor) and the 81st in the high tercile. Envelope cut by tercile
    14.6/45.8/63.7%; nearest 1.5/14.5/2.2%.
  * E_sg-leg widening (heat-load leg fixed at +2%): preference
    98.1→97.2%, pair median cut 27.5→32.2% (monotone) out to complete
    removal → bounds any differential range bias entering via the
    matching. → `paper/tables/table7_matching_construction.csv`.
- **table4 + fig2 regenerated**: fig2 now two ECDFs (nearest green
  dashed / envelope black); table4 gained nearest-match rows + the
  partner σ-spread row.
- **Manuscript** (41 pp, 0 errors): abstract/intro/conclusions lead
  with pair-level 27.5%, 35.0% labeled σ-envelope; sec 5.2 rewritten
  (three statistics + explicit non-independence sentence — pairs share
  designs, descriptive only; no binomial CI anywhere, grep-verified);
  sec 5.7 E_sg-leg sentence; sec 6: non-capture explained (det_fragile
  sample 204: γ0 −0.52° ≈3σ shallow, k_CL 1.07 → loft still airborne at
  t_max under BOTH guidance laws; brackets 29–30/300 = 9.7–10.0%, g40
  17.0–17.3%) and the "unaffected" paragraph replaced by the measured
  differential: bias factors 1.10–1.39 (det 1.25–1.33, robust 1.10–1.39,
  no front-systematic difference); knee pair truth: range ratio −2.1%
  sur → −6.6% MC, heat-load leg holds (−6.4%), σ cut grows 3.7→5.1%;
  MC SE(E[sg]) 3.5–4% at N=300 → a 2% range window is beyond any
  feasible verification budget; the heat-load leg (−3.2…+1.4%) carries
  the matching. Sec 7: range-surrogate bullet now "bounded, not
  cancelled"; 5% band stated as engineering assumption, not posterior.
- Already satisfied before this review (verified, no edit needed): 5%
  framing (sec 3.1 calibration table + Kennedy–O'Hagan wording), floor
  scoping (sec 5.1 + conclusions), seed table 3+3 (app B).

OPEN (user decision) → RESOLVED 2026-10-16: stratified matched-pair MC
RAN + folded into sec 6 (see next entry). The zero-compute bounds above
remain in the manuscript as the backstop; the measurement now leads.

### 2026-10-14 (evening) — USER GO for the pairs MC; `uq/verify_mc_pairs.py` WRITTEN + SMOKE-PASSED

Design findings while building it (both from the front CSVs):
- Tight pairs (BOTH window legs binding, |ratio| ≤ 2%: only 136 of
  5,371) have a median cut of **+0.4%** — the σ advantage is carried by
  the one-sided slack of the window (robust designs delivering less heat
  load at no less range), not by equal-performance designs differing in
  σ alone. Added as a sentence in sec 5.2 (preempt — a careful reviewer
  would find it).
- Naive selections are degenerate or uninformative: quintile-median
  NEAREST pairs cut only 0.6–5% in 4/5 quintiles (below MC resolution
  ~5–7% SE on a cut); quintile-median ENVELOPE pairs collapse onto the
  floor design (rob row 3) for 4/5 quintiles with −26…−58% heat-load
  slack (trivial match, redundant with rob_floor verification).
- **Selected design = ADVERSARIAL stress set**: among pairs with cut ≥
  the pair-median threshold (27.5%, deterministic), per E_Qs quintile
  the pair with the LEAST E_sg slack (ratio pinned at −2.0%). Selected:
  det{65,167,143,121,134} ↔ rob{72,139,142,3,75}; surrogate cuts
  +31…+67%; E_Qs ratio −26…−58% (leg verified ±3.2%, holds trivially);
  E_sg ratio at the boundary = maximally bias-exposed → if the
  differential bias breaks the comparison anywhere it breaks here
  first. Interpretation of the E_sg column = measured differential, not
  window pass/fail.
- Machinery reuses verify_mc.py (same z-LHS seed 2026 sliced to 300 →
  CRN across pair members AND shared with the 5 headline designs);
  staging uq/data/verify_mc_pairs.csv keyed (design_id, isamp),
  resume-safe, selection is a fixed 5-pair prefix (smoke rows count);
  matching.py gained a robust-row-index column 'j'. Smoke (2 sims ×
  pair 1): PASS — 4 rows staged, summary prints pair-level
  surrogate-vs-MC ratios + cut.
- LAUNCH (user, foreground+Tee, from capsule_opt, ~19–20 h):
  `C:\Users\xamii\anaconda3\python.exe -m uq.verify_mc_pairs | Tee-Object -FilePath uq\logs\verify_mc_pairs_log.txt`
  Morning after: fold results into sec 6 (one paragraph: differential
  bias measured on 5 stress pairs + cut confirmation) + recompile.
  [Status at day close: NOT yet launched — staging file holds only the
  4 smoke rows; resume-free start.]

### 2026-10-15/16 — PAIRS MC DONE + FOLDED INTO SEC 6 (Claude, on plan)

Launched by Claude 10-15 07:26 on explicit user instruction ("you launch
it"), background task bhzyi1rfa, same Tee command. **DONE 10-16 ~00:45:
2,996/2,996 (4 smoke rows reused), 1,396 min ≈ 23.3 h, 0 failures,
2,999/3,000 captured** (single non-capture = p5_det shallow lofted draw).

RESULTS (`uq/data/verify_mc_pairs{,_summary}.csv`, log
`uq/logs/verify_mc_pairs_log.txt`):
- **σ-cut CONFIRMED, the point of the run**: sur +30.9/+32.8/+49.4/
  +67.1/+30.9% → MC +38.7/+34.2/+50.1/+68.5/+32.0% (deviation ≤1.4 pp
  per pair; MC larger in 4/5). Cut survives the simulator essentially
  exactly, incl. the biggest (+67→+68.5%).
- **E_Qs ratios confirmed**: sur 0.674/0.714/0.578/0.420/0.739 vs MC
  0.681/0.707/0.575/0.402/0.738 — max deviation 4.3% (pair 4) and it
  UNDERSTATES the robust advantage.
- **E_sg differential bounded, robust-favouring**: members' bias factors
  det 1.14–1.37, rob 1.10–1.33 (all inside disclosed 1.10–1.39);
  measured pair E_sg ratios (sur matched at 0.980) = 1.044/1.002/0.948/
  0.979/1.081, mean 1.011 ± ~0.023 → matched robust designs give up no
  range in truth; det bias larger in 4/5 pairs (mean 1.264 vs 1.227).
- Floor replicates trivially (p4_rob = rob row 3 = floor design, same z
  set as verify_mc → MC sd 15.4 identical, NOT an independent draw).
- Per-design E[Qs] within −2.8/+1.4%, σ sur ≥ MC all 10 (conservative).

MANUSCRIPT (folded 10-16 morning, per plan): sec 6 gained the
stress-pair paragraph after the knee-pair evidence — selection rule,
3,000 runs, cut confirmation, E_Qs leg, measured differential + mean
ratio 1.01, "rest on direct measurement, not only on the bounds";
"Two bounds" sentence rescoped to "pairs not verified directly"; one-pair
phrase renamed "matched knee pair"; reading-guide range widened
10–30% → 9–37% across the fifteen verified designs. Recompiled:
**42 pp, 0 errors, 0 unresolved refs**; refs pp.37–38 then appendices
pp.39–40+ (order directive holds); paragraph renders p.31. paper/data
gained verification_mc_pairs_3000runs.csv + _summary.csv; paper/README
provenance updated. ALL COMPUTE FOR THE PAPER IS NOW DONE — remaining
is human: user read-through, user commit Vf4.1, submission kit,
optional geometry schematic, supervisor pass.

### 2026-10-14 (day close) — seed_check re-run DONE; review-3 round fully closed

`python -m uq.seed_check` re-run with the extended front_stats (anchors
reproduced: seed-1 nearest = 0.842 / 0.043). Per-seed nearest-match:
designs improved **68.9–84.2%**, median cut **1.9–4.3%** — the most
seed-sensitive reported statistic (single arbitrary partner per design);
conclusion (majority improved under strictest rule) holds on every seed.
Manuscript: Appendix B table gained the two nearest-match rows + prose
note; sec 5.7 seed sentence gained the clause. Recompiled: **41 pp,
0 errors, 0 unresolved refs** (includes the sec 5.2 tight-pair sentence).
Review 3 is now fully answered in the manuscript except the optional
pairs-MC armor awaiting launch above. Uncommitted delta now includes:
uq/{matching.py, verify_mc_pairs.py, seed_check.py edit, sens.py edit,
tables.py edit, figures.py edit}, paper/tables/table4+5b+7, fig2,
sections 01/05/06/07/08/10, main.tex, PLAN.md.
