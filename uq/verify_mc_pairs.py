"""Matched-pair verification MC (2026-10-14, third external review, Issue 2).

Truth-checks the matched-pair comparison on a stratified subset of pairs,
evaluated with the TRUE simulator.  The verification targets the pairs
that power the advantage statistics, in their most bias-exposed
configuration (adversarial stress set):

  * among the 5,371 eligible pairs, keep those with cut >= the pair-level
    median (27.5% -- the published statistic, the deterministic
    threshold);
  * split those by E_Qs quintile of the deterministic member; within each
    quintile select the pair with the LEAST range slack (E_sg ratio
    closest to the -2% window boundary).  If a differential range bias
    can break the comparison anywhere, it breaks here first; milder
    pairs follow a fortiori.
  * Cuts of the selected set span ~31-67%, well above the MC resolution
    of a cut (~5-7% SE at N=300).  (Two facts found while designing this:
    tight pairs -- both window legs binding, 136 of 5,371 -- have a
    median cut of +0.4%, and nearest-rule partners cut only 0.6-5%: the
    advantage lives in the one-sided slack of the window, so high-cut
    boundary pairs are the right stress set, not 'typical' pairs.)

Note: interpretation of the E_sg column is the measured differential bias
(ratio_true - ratio_surrogate), not window pass/fail -- the boundary
selection makes pass/fail meaningless by construction, and the paper's
bound is the widened-window sensitivity.

Both members of every pair are evaluated at the SAME N z-samples (the
verify_mc z-LHS, seed 2026, sliced -- common random numbers across the
pair, and shared with the five headline designs of uq/verify_mc.py), so
the pair statistics below are not polluted by sampling noise between
members.

Summary per pair (MC vs surrogate, surrogate = MomentEngine n_z=2000
seed 42, the paper lens): E_Qs ratio rob/det, E_sg ratio rob/det, sigma
cut.  Plus the per-design MC moments behind them.

Usage (from capsule_opt/):  python -m uq.verify_mc_pairs [N=300] [N_PAIRS=5]
  smoke: python -m uq.verify_mc_pairs 2 1
Timing: 10 designs x 300 sims at ~23 s  ~  19-20 h.  Resume-safe staging
to uq/data/verify_mc_pairs.csv (keyed design_id x isamp; partial runs are
valid prefixes).  Re-run the module any time to re-print the summary.
"""

import csv
import ctypes
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from capsule_opt.uncertainty.variables import sample
from uq import matching as M
from uq.verify_mc import COLS, N_MAX, SEED_MC, T_MAX, run_point

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data/verify_mc_pairs.csv')
SUMMARY = os.path.join(HERE, 'data/verify_mc_pairs_summary.csv')
N_PAIRS_DEFAULT = 5
N_QUINTILES = 5            # fixed: selection is a prefix for any npairs


def select_pairs(npairs):
    """Adversarial stratified pair selection -> [(det_row_i, rob_row_j)].

    High-cut pairs (>= pair-level median cut) at minimal range slack per
    E_Qs quintile: the configuration most exposed to a differential
    E[sg] bias.  Always built from the full 5-quintile selection, then
    truncated, so a smoke run stages rows that count toward the full run
    (same design ids).
    """
    m, _ = M.load()
    ok = m[(m.E_Qs_r <= m.E_Qs_d * (1 + M.W_QS)) &
           (m.E_sg_r >= m.E_sg_d * (1 - M.W_QS))]
    cut_thr = float(ok.cut.median())            # the published 27.5%
    adv = ok.loc[ok.cut >= cut_thr].copy()
    adv['rS'] = adv.E_sg_r / adv.E_sg_d
    q = pd.qcut(adv.E_Qs_d, N_QUINTILES, labels=False)
    pairs = []
    for k in range(N_QUINTILES):
        g = adv[q == k].sort_values(['rS', 'i', 'j'])
        row = g.iloc[0]                         # least E_sg slack
        pairs.append((int(row.i), int(row.j)))
    return adv, pairs[:npairs]


def load_designs(npairs):
    """(name, design tuple, provenance) x 2*npairs; ids 2k/2k+1 = pair k."""
    adv, pairs = select_pairs(npairs)
    det = pd.read_csv(os.path.join(HERE, 'data/opt_det_audit.csv'))
    rob = pd.read_csv(os.path.join(HERE, 'data/opt_robust.csv'))
    out = []
    for k, (i, j) in enumerate(pairs):
        r = adv[(adv.i == i) & (adv.j == j)].iloc[0]
        prov_d = (f"det row {i}: E_Qs quintile {k + 1}/{N_QUINTILES}, "
                  f"cut>=pair-median, least E_sg slack")
        prov_r = (f"rob row {j}: stress partner of det row {i} "
                  f"(sur ratios E_Qs {r.E_Qs_r / r.E_Qs_d - 1:+.1%}, "
                  f"E_sg {r.E_sg_r / r.E_sg_d - 1:+.1%}; sur cut "
                  f"{r.cut:+.1%})")
        out.append((f'p{k + 1}_det',
                    (det.rn[i], det.rs[i], det.r_theta[i]), prov_d))
        out.append((f'p{k + 1}_rob',
                    (rob.rn[j], rob.rs[j], rob.r_theta[j]), prov_r))
    return out


def summarize(npairs):
    """Pair-level MC-vs-surrogate table (+ per-design MC moments)."""
    from uq.engine import MomentEngine

    df = pd.read_csv(OUT)
    designs = load_designs(npairs)
    eng = MomentEngine()                     # n_z=2000, seed 42 (paper)
    mmt = eng.moments(np.array([list(d[1]) for d in designs]))

    mq, sq, msg, seg, nc_ = [], [], [], [], []
    print("\n===== per-design MC moments =====", flush=True)
    for i, (name, _, prov) in enumerate(designs):
        d = df[df.design_id == i]
        cap = d[(d.captured == 1) & d.Qs.notna()]
        n, nc = len(d), len(cap)
        q, sg = cap.Qs / 1e6, cap.sg / 1e3
        mq.append(q.mean()); sq.append(q.std(ddof=1))
        msg.append(sg.mean())
        nc_.append(nc)
        seg.append(sg.std(ddof=1) / math.sqrt(nc) if nc > 1 else float('nan'))
        print(f"-- {name}: N={n} captured {nc}  E[Qs] {q.mean():.1f} "
              f"(sur {mmt['E_Qs'][i] / 1e6:.1f})  sd {q.std(ddof=1):.1f} "
              f"(sur {mmt['sd_Qs_infl'][i] / 1e6:.1f})  E[sg] {sg.mean():.0f}"
              f" ± {seg[-1]:.0f} km (sur {mmt['E_sg'][i] / 1e3:.0f})",
              flush=True)
    seq = [s / math.sqrt(n) if n > 1 else float('nan')
           for s, n in zip(sq, nc_)]

    def ratio(a, b, sea, seb):
        r = a / b
        se = r * math.sqrt((sea / a) ** 2 + (seb / b) ** 2) \
            if sea and seb else float('nan')
        return r, se

    print("\n===== pair-level: does the match and the sigma cut survive "
          "the simulator? =====", flush=True)
    rows = []
    for k in range(npairs):
        i, j = 2 * k, 2 * k + 1
        sur = {'ratio_E_Qs': mmt['E_Qs'][j] / mmt['E_Qs'][i],
               'ratio_E_sg': mmt['E_sg'][j] / mmt['E_sg'][i],
               'cut': 1 - mmt['sd_Qs_infl'][j] / mmt['sd_Qs_infl'][i]}
        mc_rQ, se_rQ = ratio(mq[j], mq[i], seq[j], seq[i])
        mc_rS, se_rS = ratio(msg[j], msg[i], seg[j], seg[i])
        mc_cut = 1 - sq[j] / sq[i]
        print(f"pair {k + 1}: E_Qs ratio  sur {sur['ratio_E_Qs']:.3f}  "
              f"MC {mc_rQ:.3f} ± {se_rQ:.3f}   E_sg ratio  sur "
              f"{sur['ratio_E_sg']:.3f}  MC {mc_rS:.3f} ± {se_rS:.3f}   "
              f"sigma cut  sur {sur['cut']:+.1%}  MC {mc_cut:+.1%}",
              flush=True)
        for lab, s_, m_, se_ in (('ratio_E_Qs', sur['ratio_E_Qs'], mc_rQ, se_rQ),
                                 ('ratio_E_sg', sur['ratio_E_sg'], mc_rS, se_rS),
                                 ('sigma_cut', sur['cut'], mc_cut, float('nan'))):
            rows.append({'pair': k + 1, 'quantity': lab, 'surrogate': s_,
                         'mc': m_, 'mc_se': se_})
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(f"\nsaved {SUMMARY}", flush=True)


def main():
    if os.name == 'nt':  # keep awake while running
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    npairs = int(sys.argv[2]) if len(sys.argv) > 2 else N_PAIRS_DEFAULT
    assert n <= N_MAX, f"N <= {N_MAX} (z-LHS is fixed at N_MAX and sliced)"
    designs = load_designs(npairs)
    Z = sample(N_MAX, SEED_MC)[:n]
    k = len(designs)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline='') as f:
            for r in csv.DictReader(f):
                done.add((int(r['design_id']), int(r['isamp'])))
    todo = [(i, j) for i in range(k) for j in range(n)
            if (i, j) not in done]
    print(f"VERIFY-MC-PAIRS: {npairs} pairs ({k} designs) x {n} samples, "
          f"{len(todo)} to run (~23 s each; 5 pairs ~ 19-20 h), "
          f"t_max={T_MAX:.0f} s, z-seed {SEED_MC} (shared CRN, "
          f"same z-set as verify_mc)", flush=True)
    for i, (name, _, prov) in enumerate(designs):
        print(f"  design {i} {name}: {prov}", flush=True)

    t0 = time.time()
    newf = not os.path.exists(OUT)
    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if newf:
            w.writerow(COLS)
        for c, (i, j) in enumerate(todo, 1):
            name, design, _ = designs[i]
            w.writerow([i, j] + list(design) + list(Z[j]) +
                       run_point(design, Z[j]))
            f.flush()
            if c % 10 == 0 or c == len(todo):
                el = time.time() - t0
                print(f"  {c}/{len(todo)} done  {el / 60:.0f} min elapsed, "
                      f"ETA {(el / c * (len(todo) - c)) / 3600:.1f} h",
                      flush=True)
    summarize(npairs)


if __name__ == '__main__':
    main()
