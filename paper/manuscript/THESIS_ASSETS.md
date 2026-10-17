# Thesis asset inventory (for manuscript writing)

Source: `../../Thesis/` (one level above the repo) — original thesis-era
code + figures — and `Thesis_2003110_3.docx` (media extracted to
`../thesis_assets/docx_media/`, 19 images). Reviewed 2026-10-10.

**Style note:** all thesis figures are default-matplotlib (grid, seaborn-ish
colors, bold titles). Paper style is the converged house style (white, ticks,
no grid, no marker borders, det-blue/rob-vermillion, black frames, no
on-figure titles — captions carry them). PREFERRED ACTIONS therefore
"restyle/regenerate", using thesis figures as content/layout source. The
validation data behind them lives in `validation/reference_data/*.csv`.

## On-disk PNGs (Thesis/Capsule_optimization/)

| file | content (from visual review) | manuscript use | action |
|---|---|---|---|
| validation_dirkx/01_atmosphere.png | atmosphere vs US-76 tables, 2 panels (density, pressure vs altitude), 86–120 km extension band highlighted | Sec 2 model / appendix | restyle from US76_reference_table.csv |
| validation_dirkx/02_aero_apollo.png | Apollo CM aero coefficients vs digitized D&M data | Sec 2 validation (V2) | restyle from cp_methods_dm2017_digitized.csv + apollo_flight_wind_tunnel_data.csv |
| validation_dirkx/03_trajectory.png | Apollo entry time histories 2×2 (trimmed AoA & L/D, bank angle, stagnation heat rate, deceleration) vs D&M Fig 7.17 | Sec 2 validation (V3) — the flagship validation figure | restyle (re-run validate_apollo_end_to_end if data needed) |
| validation_dirkx/04_heating.png | stagnation + shoulder heat-flux histories (shoulder ≈ 1.5–1.6× stagnation — shoulder-heating story) | Sec 2 validation (V4a) | restyle |
| Final_result_25000/Figure_1.png | 3-D Pareto front (η_V, Qs×10⁷, sg) — thesis 25,000-eval deterministic result | Sec 5: thesis baseline front comparison | restyle from results_outputs/pareto_designs.csv; overlay with our det/robust fronts |
| Final_result_25000/Figure_2..5.png | front colored by r_Rn / r_θc / r_Rs + selected-shape views | Sec 5 background (design-space reading) | optional; fig5 already covers |
| results_outputs/01_pairwise.png | pairwise objective projections colored by (a) rn (b) rs (c) θc — thesis Fig 6.2 | optional Sec 5 | superseded by fig5 mostly |
| results_outputs/03_silhouettes.png | Dirkx-style capsule meridional silhouettes, 5 selected designs + Apollo | **HIGH VALUE**: regenerate with OUR det-knee / robust-knee / robust-matched + Apollo reference, house style → new Sec 5 figure | regenerate (geometry code in repo) |

## Docx media (paper/thesis_assets/docx_media/, image1..19)

Captions partially mapped (Fig 3.4 atmosphere, Fig 3.5 **capsule with local
normal — geometry schematic, good Sec 2 candidate**, Fig 4.2 trajectory
profile, Fig 5.1 single-evaluation trace, Fig 6.1/6.2 front + pairwise).
Identify the rest by viewing during drafting; schematics (geometry, flow,
normal vectors) can be used as-is or redrawn — data plots should be
restyled from CSVs instead.

## Other usable material

| item | content | use |
|---|---|---|
| Final_result_25000/Final_sim.txt + Final_Simulation result.txt | final thesis design trajectory histories (text) | optional "recommended design trajectory" panel (we can also simulate our robust knee directly) |
| results_outputs/pareto_designs.csv | thesis Pareto designs | Sec 5 baseline overlay data |
| results_outputs/02_correlations.txt | objective/design correlations | discussion nuggets |
| capsule_optimization.ipynb (×2) | exploration notebook w/ outputs | mine on demand |
| Thesis Table 6.2 (2930–4695 km sg envelope) | in docx | already cited in sanity envelope check |

## Top 5 figures to (re)build for the paper from thesis material

1. Capsule silhouettes: Apollo + det-knee + robust-knee + robust-matched
   (from 03_silhouettes layout, house style) — Sec 5
2. Apollo trajectory validation 2×2 (03_trajectory, restyled) — Sec 2
3. Apollo aero coefficients validation (02_aero_apollo, restyled) — Sec 2
4. Thesis deterministic front vs paper fronts overlay (Figure_1 +
   pareto_designs.csv + our fronts, restyled 2-D projections not 3-D) — Sec 5
5. Geometry schematic (docx Fig 3.5 style; capsule + local normal +
   Rn/Rs/θc annotations) — Sec 2
