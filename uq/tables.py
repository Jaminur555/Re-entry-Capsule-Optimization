"""Manuscript results tables (Step 6) -> paper/tables/.

Reads the Step-6 result CSVs from uq/data/ (pipeline location) and writes
curated long-form tables for the paper:

  table1_knee_designs.csv    knee-point designs, physical units, robust lens
                             (same quantities as uq.knee's printed table)
  table2_verification_mc.csv copy of the verify_mc summary (MC vs surrogate)
  table3_det_front_audit.csv det-front fragility audit headline statistics
  table4_matched_pairs.csv   matched-performance det-vs-robust comparison

Usage: python -m uq.tables     (seconds; run after knee + audit + verify_mc)
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TAB = os.path.join(os.path.dirname(HERE), 'paper', 'tables')

# quantity label -> (knee_designs column, unit, scale factor)
T1 = [
    ('Nose radius Rn', 'Rn_m', 'm', 1.0),
    ('Shoulder radius Rs', 'Rs_m', 'm', 1.0),
    ('Afterbody cone angle theta_c', 'theta_c_deg', 'deg', 1.0),
    ('Volumetric efficiency eta_V', 'eta_V', '-', 1.0),
    ('Heat load Qs at z0 (simulator)', 'Qs_z0_MJ', 'MJ/m2', 1.0),
    ('Skip range sg at z0 (simulator)', 'sg_z0_km', 'km', 1.0),
    ('Peak stag. heating q_stag at z0', 'q_stag_z0_kW', 'kW/m2', 1.0),
    ('Peak shldr heating q_shldr at z0', 'q_shldr_z0_kW', 'kW/m2', 1.0),
    ('Peak load n_max at z0', 'n_max_z0_g', 'g', 1.0),
    ('E[Qs]', 'E_Qs_MJ', 'MJ/m2', 1.0),
    ('sigma_Qs (inflated)', 'sd_Qs_MJ', 'MJ/m2', 1.0),
    ('CV(Qs)', 'CV_Qs', '%', 100.0),
    ('E[sg]', 'E_sg_km', 'km', 1.0),
    ('q_stag E+2sigma', 'q_stag_max_E2S', 'kW/m2', 1.0),
    ('q_shldr E+2sigma', 'q_shldr_max_E2S', 'kW/m2', 1.0),
    ('n_max E+2sigma', 'n_max_E2S', 'g', 1.0),
    ('P(q_stag > limit)', 'Pexc_q_stag_max', '%', 100.0),
    ('P(q_shldr > limit)', 'Pexc_q_shldr_max', '%', 100.0),
    ('P(n_max > limit)', 'Pexc_n_max', '%', 100.0),
    ('P(slow family)', 'P_slow', '%', 100.0),
    ('sigma_Qs cut vs det knee', 'sd_cut_pct', '%', 1.0),
]


def table1():
    df = pd.read_csv(os.path.join(DATA, 'knee_designs.csv'))
    cols = ['det_knee', 'robust_knee', 'robust_matched']
    by = {r['mode']: r for _, r in df.iterrows()}
    key = {'det_knee': 'det', 'robust_knee': 'robust',
           'robust_matched': 'robust_m'}
    rows = []
    for lab, c, u, s in T1:
        r = {'quantity': lab, 'unit': u}
        for k in cols:
            v = by[key[k]].get(c)
            r[k] = round(float(v) * s, 4) if pd.notna(v) else ''
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'table1_knee_designs.csv'),
                              index=False)


def table2():
    src = pd.read_csv(os.path.join(DATA, 'verify_mc_summary.csv'))
    src.to_csv(os.path.join(TAB, 'table2_verification_mc.csv'), index=False)


def table3():
    from uq.optimize import LIMITS
    d = pd.read_csv(os.path.join(DATA, 'opt_det_audit.csv'))
    sd, e = d.aud_sd_Qs_infl, d.aud_E_Qs
    rows = [
        ('sigma_Qs median', np.median(sd) / 1e6, 'MJ/m2'),
        ('sigma_Qs range', f'{sd.min() / 1e6:.1f} - {sd.max() / 1e6:.1f}',
         'MJ/m2'),
        ('CV(Qs) median', np.median(sd / e), '-'),
    ]
    for o in ('q_stag_max', 'q_shldr_max', 'n_max'):
        pe = d[f'aud_Pexc_{o}']
        rows += [
            (f'P({o} > limit) median', np.median(pe), '-'),
            (f'P({o} > limit) max', pe.max(), '-'),
            (f'fraction P({o}) > 1%', (pe > 0.01).mean(), '-'),
        ]
    ps = d.aud_P_slow
    rows += [
        ('P(slow) median', np.median(ps), '-'),
        ('P(slow) range', f'{ps.min():.3f} - {ps.max():.3f}', '-'),
    ]
    rf = np.ones(len(d), bool)
    for o, L in LIMITS.items():
        rf &= (d[f'aud_E_{o}'] + 2 * d[f'aud_sd_{o}_infl']) <= L
    rows.append(('robust-feasible fraction of det front', rf.mean(), '-'))
    pd.DataFrame(rows, columns=['statistic', 'value', 'unit']).to_csv(
        os.path.join(TAB, 'table3_det_front_audit.csv'), index=False)


def table4():
    from uq.matching import W_QS, load as load_pairs
    d = pd.read_csv(os.path.join(DATA, 'opt_det_audit.csv'))
    r = pd.read_csv(os.path.join(DATA, 'opt_robust.csv'))
    m, n_det = load_pairs()
    ok = m[(m.E_Qs_r <= m.E_Qs_d * (1 + W_QS)) &
           (m.E_sg_r >= m.E_sg_d * (1 - W_QS))]
    best = ok.groupby('i')['cut'].max()
    near = ok.loc[ok.groupby('i')['dist'].idxmin()]
    spread = ok.groupby('i')['sd_r'].agg(lambda s: s.max() / s.min())
    rows = [
        ('det designs with >= 1 matched robust design',
         f'{ok.i.nunique()} / {n_det}', '-'),
        ('matched (det, robust) pairs', len(ok), '-'),
        ('PAIR-level: fraction of pairs with sigma_robust < sigma_det',
         (ok.cut > 0).mean(), '-'),
        ('PAIR-level: median sigma cut over all pairs', ok.cut.median(), '-'),
        ('DESIGN-level: fraction of det designs beaten by best match',
         (best > 0).mean(), '-'),
        ('DESIGN-level: median best-match sigma cut (sigma-envelope; '
         'fig2 ECDF median)', best.median(), '-'),
        ('DESIGN-level: fraction beaten by nearest match (prespecified)',
         (near.cut > 0).mean(), '-'),
        ('DESIGN-level: median nearest-match sigma cut (fig2 ECDF median)',
         near.cut.median(), '-'),
        ('median sigma spread of eligible partners (max/min)',
         spread.median(), '-'),
        ('worst-member P(q_stag) -- det front', d.aud_Pexc_q_stag_max.max(),
         '-'),
        ('worst-member P(q_stag) -- robust front', r.Pexc_q_stag_max.max(),
         '-'),
        ('worst-member P(q_shldr) -- det front', d.aud_Pexc_q_shldr_max.max(),
         '-'),
        ('worst-member P(q_shldr) -- robust front',
         r.Pexc_q_shldr_max.max(), '-'),
    ]
    pd.DataFrame(rows, columns=['statistic', 'value', 'unit']).to_csv(
        os.path.join(TAB, 'table4_matched_pairs.csv'), index=False)


def main():
    os.makedirs(TAB, exist_ok=True)
    table1()
    table2()
    table3()
    table4()
    for f in sorted(os.listdir(TAB)):
        print(f"  {f}", flush=True)


if __name__ == '__main__':
    main()
