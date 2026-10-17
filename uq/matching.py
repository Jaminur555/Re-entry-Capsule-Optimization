"""Matching-construction audit (2026-10-14, third external review).

Two reviewer asks, both answerable from the saved front CSVs with no
simulator and no engine re-run (the audit/report columns ARE the baseline
CRN moments at n_z=2000):

1. A prespecified matching rule that does not select on the outcome.  The
   published design-level statistic takes, per deterministic design, the
   ELIGIBLE robust design with the lowest sigma (largest cut) -- selection
   on the outcome, optimistic by construction.  Here we add the
   NEAREST-eligible match: among partners inside the window, the one
   closest in relative performance space,
       d = sqrt( (E_Qs_r/E_Qs_d - 1)^2 + (E_sg_r/E_sg_d - 1)^2 ),
   ties broken by the lowest robust-front index.  The rule is fixed before
   any outcome is examined; only the matching variables enter it.

2. Does the robust advantage survive the surrogate's E[sg] bias?  The bias
   is design-specific (factors 1.10-1.39 across the five MC-verified
   designs), so the E[sg] leg of the window cannot be certified in truth
   at 2%.  Bound instead by widening ONLY that leg (E_Qs window fixed at
   +2%): if the pair statistics are stable out to windows far larger than
   any plausible differential bias, the conclusion does not ride on it.

Anchors: at the base window this must reproduce the published numbers
(5,371 pairs / 98.1% / 27.5% pair median / 35.0% best-match median).

Output: paper/tables/table7_matching_construction.csv + console table.

Usage:  python -m uq.matching     (seconds)
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TAB = os.path.join(os.path.dirname(HERE), 'paper', 'tables')

W_QS = 0.02            # heat-load window (simulator-verified leg), fixed


def load():
    d = pd.read_csv(os.path.join(DATA, 'opt_det_audit.csv'))
    r = pd.read_csv(os.path.join(DATA, 'opt_robust.csv'))
    dd = d[['aud_E_Qs', 'aud_E_sg', 'aud_sd_Qs_infl']].reset_index(
        names='i').rename(columns={'aud_E_Qs': 'E_Qs_d',
                                   'aud_E_sg': 'E_sg_d',
                                   'aud_sd_Qs_infl': 'sd_d'})
    rr = r[['E_Qs', 'E_sg', 'sd_Qs_infl']].reset_index(
        names='j').rename(columns={'E_Qs': 'E_Qs_r', 'E_sg': 'E_sg_r',
                                   'sd_Qs_infl': 'sd_r'}).assign(k=1)
    m = dd.assign(k=1).merge(rr, on='k')
    m['cut'] = 1 - m.sd_r / m.sd_d
    m['dist'] = np.sqrt((m.E_Qs_r / m.E_Qs_d - 1) ** 2 +
                        (m.E_sg_r / m.E_sg_d - 1) ** 2)
    return m, len(d)


def stats(m, n_det, w_sg):
    """All matching statistics for one E[sg] window width."""
    ok = m[(m.E_Qs_r <= m.E_Qs_d * (1 + W_QS)) &
           (m.E_sg_r >= m.E_sg_d * (1 - w_sg))]
    if ok.empty:
        return {}
    best = ok.groupby('i')['cut'].max()                # outcome-selective
    near = ok.loc[ok.groupby('i')['dist'].idxmin()]    # prespecified rule
    return {
        'matched pairs': len(ok),
        'pair frac favour robust': (ok.cut > 0).mean(),
        'pair median cut': ok.cut.median(),
        'det designs matched': int(ok.i.nunique()),
        'best-match frac beaten': (best > 0).mean(),
        'best-match median cut': best.median(),
        'nearest-match frac beaten': (near.cut > 0).mean(),
        'nearest-match median cut': near.cut.median(),
    }


def main():
    m, n_det = load()
    rows = []
    for w in (0.02, 0.05, 0.10, 0.15, 1.00):
        st = stats(m, n_det, w)
        for k, v in st.items():
            rows.append({'E_sg window': f'{w:.0%}', 'statistic': k,
                         'value': round(float(v), 4)})
        print(f"[w_sg={w:.0%}] " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in st.items()), flush=True)

    os.makedirs(TAB, exist_ok=True)
    out = os.path.join(TAB, 'table7_matching_construction.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten: {out}", flush=True)


if __name__ == '__main__':
    main()
