"""Sanity checks for the surrogate stack (PLAN.md Step 5; v2-mixture update
2026-10-10: gp_sg.joblib is gone -- sg comes from the MomentEngine mixture).

1. Corner check: surrogate (MomentEngine.det_predict at nominal z0) vs the
   true simulator at 10 random design points (t_max=14400, captured points
   only; non-captured points are skipped and counted). Deviations
   |surrogate - direct| / DOE range vs the final-gate RMSE bands (n=2800,
   CV_RESTARTS=2): Qs 2.33%, q_stag 2.29%, q_shldr 3.01%, n_max 2.23%.
   sg is checked BRANCH-MATCHED (realized fast/slow family vs its own GP
   branch; bands 3.8% fast, 15% slow-irreducible) -- the mixture mean is
   also reported for reference but cannot match a single realization
   (structural branch gap; E[sg] bias +9-28% verified by uq.verify_mc).
2. Envelope check: DOE sg range vs thesis Table 6.2 (2930-4695 km).

Usage: python -m uq.sanity_check      (10 simulator runs, ~3-5 min)
"""

import os

import numpy as np
import pandas as pd

from capsule_opt.optimization.objectives import evaluate_shape
from uq.engine import MomentEngine

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = {'Qs': 0.02332, 'q_stag_max': 0.02294, 'q_shldr_max': 0.03009,
        'n_max': 0.02225}
N_PTS = 10
SLOW_CUT = 2250.0


def main():
    eng = MomentEngine()
    doe = pd.read_csv(os.path.join(HERE, 'data/doe_results.csv'))
    rng = {o: doe[o].max() - doe[o].min()
           for o in list(GATE) + ['sg']}

    # --- 1. corner check at nominal z --------------------------------------
    d = np.random.RandomState(7).uniform(size=(N_PTS, 3))
    direct, X, t_fin = [], [], []
    skipped = 0
    for i, (rn, rs, rt) in enumerate(d):
        r = evaluate_shape(rn, rs, rt, aero_scales=(1.0, 1.0),
                           entry_conditions={'gamma0_deg': -2.0,
                                             'Vo': 7830.0, 't_max': 14400.0})
        if 'termination event' not in r['traj']['status']:
            skipped += 1
            print(f"  point {i + 1}/{N_PTS}: not captured, skipped",
                  flush=True)
            continue
        det = r['constraints']['details']
        direct.append({'Qs': r['objectives']['Qs'], 'sg': r['objectives']['sg'],
                       'q_stag_max': det['q_stag_max'],
                       'q_shldr_max': det['q_shldr_max'],
                       'n_max': det['n_max']})
        X.append((rn, rs, rt))
        t_fin.append(r['traj']['t'][-1])
        print(f"  point {i + 1}/{N_PTS}: t_final {t_fin[-1]:.0f} s "
              f"({'slow' if t_fin[-1] > SLOW_CUT else 'fast'})", flush=True)
    direct = pd.DataFrame(direct)
    Xd = np.array(X)
    sur = eng.det_predict(Xd)
    n = len(direct)

    print(f"\ncorner check (|surrogate-direct| / DOE range, {n} pts, "
          f"{skipped} skipped):")
    print(f"{'output':<12}{'median':>9}{'max':>9}{'band':>7}  verdict")
    for o, band in GATE.items():
        a = (sur[o][:n] - direct[o].to_numpy()) / rng[o]
        a = np.abs(a)
        ok = np.sqrt((a ** 2).mean()) < band    # RMSE-level criterion
        print(f"{o:<12}{np.median(a):>9.3%}{a.max():>9.3%}{band:>7.0%}"
              f"  {'PASS' if ok else 'FAIL'}")
    # sg: a realization is EITHER fast or slow -- the mixture mean cannot
    # match one branch, so compare BRANCH-MATCHED (realized family vs its
    # own GP branch; mixture bands: fast 3.8%, slow ~15% irreducible).
    from uq.engine import _log_moments
    X7 = np.hstack([Xd, np.tile([-2.0, 7830.0, 1.0, 1.0], (n, 1))])
    sc = eng.mix['scaler']
    mu_f, sd_f = eng._predict_log({'gp': eng.mix['gp_fast'], 'scaler': sc},
                                  X7)
    mu_s, sd_s = eng._predict_log({'gp': eng.mix['gp_slow'], 'scaler': sc},
                                  X7)
    m_f, _ = _log_moments(mu_f, sd_f)
    m_s, _ = _log_moments(mu_s, sd_s)
    fast = np.array([t <= SLOW_CUT for t in t_fin])
    for lab, pred, m_, band in (('sg-fast', m_f, fast, 0.038),
                                ('sg-slow', m_s, ~fast, 0.15)):
        if m_.sum():
            a = np.abs((pred[m_] - direct['sg'].to_numpy()[m_]) / rng['sg'])
            ok = np.sqrt((a ** 2).mean()) < band
            print(f"{lab:<12}{np.median(a):>9.3%}{a.max():>9.3%}"
                  f"{band:>7.0%}  {'PASS' if ok else 'FAIL'}  "
                  f"({int(m_.sum())} pts)")
    a = np.abs((sur['sg'][:n] - direct['sg'].to_numpy()) / rng['sg'])
    print(f"{'sg-mix':<12}{np.median(a):>9.3%}{a.max():>9.3%}"
          f"{'--':>7}  n/a (mean vs realization: structural branch gap;"
          f" E[sg] bias verified by verify_mc: +9-28%)")

    # --- 2. envelope check vs thesis Table 6.2 -----------------------------
    feas = doe[doe['feasible'] > 0]
    t62 = (2930.0, 4695.0)
    lo, hi = feas['sg'].min() / 1e3, feas['sg'].max() / 1e3
    med = feas['sg'].median() / 1e3
    print(f"\nenvelope: feasible-DOE sg [{lo:.0f}, {hi:.0f}] km "
          f"(median {med:.0f}) vs thesis Table 6.2 {t62} km")
    print("contains thesis range" if lo <= t62[0] and hi >= t62[1]
          else "DOES NOT contain thesis range")


if __name__ == '__main__':
    main()
