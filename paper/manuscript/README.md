# Manuscript (LaTeX, Acta Astronautica / elsarticle) — sectionized

`main.tex` = preamble + Acta frontmatter + `\input` chain + **References
last**. Section content lives in `sections/`:

```
manuscript/
  main.tex               preamble, frontmatter, \input chain, bibliography
  refs.bib               26 entries, ALL verified (PDFs, thesis
                         bibliography, Crossref, JSST journal page).
                         ress2025rbmdo REMOVED 10-13: phantom (no such
                         RESS paper exists). hashemi2019capsule =
                         Ghaedamini Harouni & Hashemi Mehneh, JSST
                         12(4):1-17, 2019 (robust capsule MDO, no
                         det-vs-robust comparison).
  THESIS_ASSETS.md       thesis-figure inventory + section mapping
  sections/
    01_introduction.tex        full prose (context/prior work/gaps/C1-C4)
    02_vehicle_model.tex       geometry/aero/trajectory + validation w/
                               claims-dependency paragraph
    03_uncertainty_surrogate.tex  UQ model, DOE, GPs, gate table
    04_formulation.tex         det/robust equations, CRN engine, figs
    05_results.tex             figs 1-5, tables: matched/audit/knees/sens
                               + Sec 5.7 sensitivity (uq/sens.py)
    06_verification.tex        MC verification table + E[sg] disclosure
    07_discussion.tex          design-practice discussion + limitations
    08_conclusions.tex         written (w/ sensitivity sentence)
    09_appendix_validation.tex Appendix A: identity table, variant family,
                               deviations, case identity (DEVIATIONS.md)
```

## Compile (verified working; MiKTeX present locally)

pdflatex -interaction=nonstopmode main && bibtex main &&
pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

Last verified: 11 pages, 17/17 bib entries, all citations/references
resolved, no errors.

## What's live vs TODO

- LIVE (2026-10-12): FULL PROSE DRAFT in all sections 01-08 (26 pp, 0
  errors, References last). Equations (det + constraints, robust,
  sigma_infl), figs 1-6 + figS1 with locked-number captions, 7 booktabs
  tables (bounds / uncertainty / gate / matched / audit / knees /
  verification), all 17 refs cited live.
- refs.bib: 15 of 17 entries VERIFIED against source PDFs
  (../../references/) and the thesis bibliography. Corrections made:
  book title = "Conceptual Shape Optimization of Entry Vehicles"; TN
  D-3890 = Moseley/Moore/Hughes 1967 (key moseley1967d3890); Ridolfi =
  AIAA 2012-5006 conference paper (not a thesis); Zoby-Sullivan = NASA
  TM X-1067. STILL [VERIFY]: ress2025rbmdo, hashemi2019capsule (no PDFs
  on disk; web search could not confirm).
- TODO: geometry schematic figure (the one thesis figure left, needs a
  visual feedback round) or drop it; submission kit (highlights,
  declarations, CRediT, data-availability statement, cover letter with
  thesis-derived note); Elsevier AI-policy disclosure decision
  (user + supervisor); supervisor pass. Everything else is DONE:
  full prose 01-08, Appendix A (validation forensics) + Appendix B
  (3+3 seed repeatability), Sec 5.7 sensitivity table, Sec 6.2
  guidance-law sensitivity, 26/26 verified references, author block,
  12 live figures.

## LaTeX gotchas learned here (don't repeat)

- `%` comments are ILLEGAL inside BibTeX entries — keep [VERIFY] markers
  inside field values (they render in the PDF as fill-me reminders).
- `\cite` inside `%`-commented lines never reaches the .aux — citation
  anchor sentences must be live text.

## Locked-number conventions

98.1% pair-level / 35.0% design-level median sigma cut (never the obsolete
43.5%); sigma floor 16.4 MJ/m2 (MC 15.4); det-fragile P_exc(q_shldr) 9.7%
vs 10.0% promised; E[sg] bias +9-28% disclosed; units via siunitx.
