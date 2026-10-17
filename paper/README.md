# Paper assembly area (manuscript assets — robust capsule design, Acta Astronautica)

Frozen, descriptively-named copies of everything the manuscript cites. The
working pipeline lives in `uq/` (scripts + regenerating data); do not edit
files here — regenerate them (map below) so provenance stays one-directional:
`uq/` scripts → `paper/` assets.

## figures/   (regenerate: `python -m uq.figures`)

| file | content |
|---|---|
| fig1_front_sigma.{png,pdf} | det vs robust fronts, σ_Qs vs E[Qs], irreducible floor line |
| fig2_matched_sigma_cut.{png,pdf} | ECDF of best-match σ cut per det design (median 35.0%) |
| fig3_violation_ecdf.{png,pdf} | P(q > limit) ECDFs, both fronts |
| fig4_front_projections.{png,pdf} | η_V and E[sg] vs E[Qs] projections |
| fig5_design_space_sigma_map.{png,pdf} | design-space σ map + both fronts |
| fig6_probe_true_sigma.{png,pdf} | probe: true σ_z per output, 5 fixed designs |

## data/   (frozen copies of `uq/data/` — source of every number)

| file | content |
|---|---|
| doe_simulator_dataset_2800runs.csv | DOE backing the surrogates |
| det_front_with_robust_audit.csv | det front + audited robust view (n_z=2000) |
| robust_front.csv | robust front (n_z=2000 report columns) |
| knee_designs.csv | the 3 knee-table designs (all columns) |
| knee_sim_z0.csv | simulator-at-z₀ cache for those designs |
| verification_mc_1500runs.csv | true-simulator MC, 5 designs × 300 |
| verification_mc_summary.csv | MC vs surrogate comparison (incl. E[sg] bias) |
| verification_mc_pairs_3000runs.csv | true-simulator MC, 5 adversarial matched pairs × 2 members × 300 |
| verification_mc_pairs_summary.csv | pair-level sur-vs-MC: E_Qs/E_sg ratios + σ-cuts (sec 6 stress-pair paragraph) |
| probe_true_dispersion_300runs.csv | probe-0 (σ_z vs gate-RMSE verdict) |
| design_space_scan_300pts.csv | 300-pt feasibility scan (fig5 background) |

## tables/   (regenerate: `python -m uq.tables`)

| file | content |
|---|---|
| table1_knee_designs.csv | knee designs, physical units, robust lens |
| table2_verification_mc.csv | verification summary; E[Qs]/σ/peaks verified, E[sg] biased +9–28% |
| table3_det_front_audit.csv | det-front fragility headline stats |
| table4_matched_pairs.csv | matched-pair comparison — pair-level AND design-level statistics |

## supplementary/

- fig1_interactive_front.html — interactive companion of fig1
- figS1_nsga2_convergence.{png,pdf} — NSGA-II convergence evidence

## manuscript/   (LaTeX, Acta Astronautica / elsarticle)

`main.tex` + `refs.bib` + `sections/01..08_*.tex` — sectionized skeleton;
each section's comment block is its content plan with the locked numbers.
See `manuscript/README.md`. Write order: intro/differentiation first.

## thesis_assets/

`docx_media/` — 19 figures extracted from the thesis document (originals:
`../../Thesis/Thesis_2003110_3.docx`, outside the repo). Inventory + mapping
to manuscript sections: `manuscript/THESIS_ASSETS.md`.

## Not in this folder

Phase-1 validation material stays at repo locations: `validation/`
(+ DEVIATIONS.md), `validation_data/`, `references/`, `results/` (Phase-1
figures/json). Plan and decision log: `uq/PLAN.md`. Manuscript text
sections: to be added here during writing weeks.
