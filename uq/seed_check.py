"""NSGA-II seed repeatability (Step-6 hardening, 2026-10-10; extended
2026-10-12 for the robust front per external review).

Two independent comparisons:
  det mode     opt_det.csv (seed 1) vs opt_det_s2/s3.csv
               F = (-eta_V, Qs, -sg); feasibility from point g_* columns.
  robust mode  opt_robust.csv (seed 1) vs opt_robust_s2/s3.csv
               F = (-eta_V, E_Qs, -E_sg, sd_Qs_infl); feasibility from the
               robust problem's g_* columns (already E+2sigma form).
               Robust seeds that are still missing are skipped, so this
               runs safely before the Phase-B nights land.

For each mode: per-seed objective statistics, merged non-dominated-set
composition, pairwise cross-domination. Robust mode additionally recomputes
the headline matched-pair statistics (uq.sens.front_stats, same lens as
table4/table6: det front audited through the moment engine at n_z=2000
seed 42, each robust seed native) -> per-seed pairs / fraction favouring
robust / median cuts, so the headline numbers get seed error bars.

Usage: python -m uq.seed_check     (det: seconds; robust: ~8 min/seed of
                                    surrogate moment evaluation)
Writes paper/tables/table5_det_seed_repeatability.csv and
            paper/tables/table5b_robust_seed_repeatability.csv
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(HERE), 'paper', 'tables')

MODES = {
    'det': dict(
        seeds=[('seed 1', 'opt_det.csv'), ('seed 2', 'opt_det_s2.csv'),
               ('seed 3', 'opt_det_s3.csv')],
        obj={'eta_V': 'max', 'Qs': 'min', 'sg': 'max'},
        out='table5_det_seed_repeatability.csv'),
    'robust': dict(
        seeds=[('seed 1', 'opt_robust.csv'), ('seed 2', 'opt_robust_s2.csv'),
               ('seed 3', 'opt_robust_s3.csv')],
        obj={'eta_V': 'max', 'E_Qs': 'min', 'E_sg': 'max',
             'sd_Qs_infl': 'min'},
        out='table5b_robust_seed_repeatability.csv'),
}
G_COLS = ['g_q_stag', 'g_q_shldr', 'g_n_max', 'g_pitch', 'g_nose']


def F_of(df, obj):
    """Objective matrix, all minimized: flip the maximized columns."""
    cols, sign = zip(*[(o, -1.0 if s == 'max' else 1.0)
                       for o, s in obj.items()])
    return df[list(cols)].to_numpy() * np.array(sign)


def dominated_frac(Fa, Fb):
    """Fraction of a-points weakly dominated (strictly once) by some b-point."""
    hit = np.zeros(len(Fa), bool)
    for i, f in enumerate(Fa):
        hit[i] = ((Fb <= f).all(axis=1) & (Fb < f).any(axis=1)).any()
    return hit.mean()


def run_mode(mode, cfg):
    fronts = {}
    for lab, fn in cfg['seeds']:
        p = os.path.join(HERE, 'data', fn)
        if os.path.exists(p):
            fronts[lab] = pd.read_csv(p)
        else:
            print(f"  skip {fn} (not found)", flush=True)
    if not fronts:
        print(f"[{mode}] no fronts found", flush=True)
        return

    obj = cfg['obj']
    rows = []
    print(f"\n[{mode}] {'seed':<8}{'n':>4}{'feas':>7}   objectives "
          "(min / median / max)", flush=True)
    for lab, df in fronts.items():
        feas = (df[G_COLS] <= 0).all(axis=1).mean()
        s = [f"{o} {df[o].min():.4g}/{df[o].median():.4g}/{df[o].max():.4g}"
             for o in obj]
        print(f"[{mode}] {lab:<8}{len(df):>4}{feas:>7.0%}   " +
              "  ".join(s), flush=True)
        for o in obj:
            rows.append({'seed': lab, 'quantity': o, 'min': df[o].min(),
                         'median': df[o].median(), 'max': df[o].max()})

    labs = list(fronts)
    if len(labs) >= 2:
        allF = np.vstack([F_of(fronts[l], obj) for l in labs])
        keep = np.ones(len(allF), bool)
        for i in range(len(allF)):
            keep[i] = not ((allF <= allF[i]).all(1) &
                           (allF < allF[i]).any(1)).any()
        comp, i0 = [], 0
        for l in labs:
            n = len(fronts[l])
            comp.append(int(keep[i0:i0 + n].sum()))
            i0 += n
        print(f"[{mode}] merged non-dominated set: {int(keep.sum())} of "
              f"{len(allF)} | contributions " +
              " / ".join(f"{l}: {c}" for l, c in zip(labs, comp)),
              flush=True)
        for i, l in enumerate(labs):
            rows.append({'seed': 'merged', 'quantity': f'nd_from_{l}',
                         'min': float('nan'), 'median': float('nan'),
                         'max': comp[i]})
        print(f"[{mode}] pairwise cross-domination (row front dominated by "
              "col front):", flush=True)
        print(f"{'':<10}" + "".join(f"{l:>10}" for l in labs), flush=True)
        for la in labs:
            fr = [f"{dominated_frac(F_of(fronts[la], obj),
                                    F_of(fronts[lb], obj)):>10.1%}"
                  for lb in labs]
            print(f"{la:<10}" + "".join(fr), flush=True)
            for lb in labs:
                rows.append({'seed': la, 'quantity': f'dominated_by_{lb}',
                             'min': float('nan'), 'median': float('nan'),
                             'max': dominated_frac(F_of(fronts[la], obj),
                                                   F_of(fronts[lb], obj))})

    out = os.path.join(TAB, cfg['out'])
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{mode}] saved {out}", flush=True)


def robust_matched():
    """Headline matched statistics per robust seed (review ask E)."""
    from uq.engine import MomentEngine
    from uq.sens import front_stats

    det = pd.read_csv(os.path.join(HERE, 'data/opt_det.csv'))
    Xd = det[['rn', 'rs', 'r_theta']].to_numpy()
    eng = MomentEngine()                     # n_z=2000, seed 42 (paper lens)
    mD = eng.moments(Xd)
    rows = []
    for lab, fn in MODES['robust']['seeds']:
        p = os.path.join(HERE, 'data', fn)
        if not os.path.exists(p):
            continue
        mR = eng.moments(pd.read_csv(p)[['rn', 'rs', 'r_theta']].to_numpy())
        st = front_stats(mD, mR)
        keep = ('floor', 'matched_pairs', 'pair_frac', 'pair_median',
                'design_matched', 'design_frac', 'design_median', 'nearest')
        sel = {k: v for k, v in st.items() if k.startswith(keep)}
        print(f"[matched vs {lab}] " + "  ".join(
            f"{k.split('_')[0]}={v:,.3f}" for k, v in sel.items()),
            flush=True)
        for k, v in sel.items():
            rows.append({'robust_seed': lab, 'statistic': k,
                         'value': v / 1e6 if 'MJ/m2' in k else v})
    if rows:
        out = os.path.join(TAB, 'table5b_robust_matched_by_seed.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[matched] saved {out}", flush=True)


def main():
    os.makedirs(TAB, exist_ok=True)
    run_mode('det', MODES['det'])
    run_mode('robust', MODES['robust'])
    robust_matched()


if __name__ == '__main__':
    main()
