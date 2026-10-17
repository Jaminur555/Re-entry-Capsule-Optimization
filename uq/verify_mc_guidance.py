"""Night-3 revision run: guidance-law sensitivity of the verification MC.

Reviewer question (Phase-A plan, 2026-10-12): do the paper's verified
conclusions survive a materially different guidance realization?  Re-runs
the SAME 5 verification designs x 300 z-samples of uq/verify_mc.py under
the alternate no-skip bank law "Eq. 2.40 as printed" (sin-gamma Coriolis
term, 82-deg command cap) -- the variant that reproduces the reference
track length to -1.1% (Appendix A.2 / validation variant family).  Both
laws are plausible readings of the same reference guidance spec, so this
brackets the guidance contribution to the verified statistics.

Stage/resume: uq/data/verify_mc_g40.csv (same columns/keying as
verify_mc.csv).  After staging, the summary compares MC under the base law
(uq/data/verify_mc.csv) against MC under the alternate law, per design:
E/sd[Qs], E[sg], P_slow, peak exceedances.  Surrogate references in the
staged console output refer to the BASE-law-trained surrogate and are
expected to be biased here -- the guidance comparison is MC-vs-MC.

Usage (from capsule_opt/):  python -m uq.verify_mc_guidance [N=300] [K=5]
  smoke: python -m uq.verify_mc_guidance 2 1
Timing: as verify_mc, ~15-50 s per sim -> 5 x 300 ~ 10-12 h.
"""

import csv
import os
import sys

import numpy as np
import pandas as pd

import capsule_opt.trajectory.trajectory as tt
import uq.verify_mc as V
from validation import apollo_case
from validation.system.sensitivity_bank_cl import make_gated_bank

# apollo_case FIXES config.Rm_fixed/Lc_fixed at Apollo values (1.956/2.662)
# for the whole process on import -- correct for the Phase-1 validation
# scripts, WRONG for the study geometry (Rm=2, Lc=2).  Restore immediately:
# every staged sim and the summary must run at the study defaults.
apollo_case.restore_defaults()
assert tt.config.Rm_fixed == 2.0 and tt.config.Lc_fixed == 2.0, \
    'study geometry must be restored before running'

HERE = os.path.dirname(os.path.abspath(__file__))
G40 = os.path.join(HERE, 'data/verify_mc_g40.csv')
G40_SUM = os.path.join(HERE, 'data/verify_mc_g40_summary.csv')
BASE = os.path.join(HERE, 'data/verify_mc.csv')
ORIGINAL_BANK = tt.bank_angle_sigma


def compare(k):
    """MC(base law) vs MC(alternate law) per design -> summary CSV."""
    b = pd.read_csv(BASE)
    g = pd.read_csv(G40)

    def stats(df, i):
        d = df[df.design_id == i]
        cap = d[(d.captured == 1) & d.Qs.notna()]
        return {
            'N': len(d), 'captured': len(cap),
            'E_Qs': cap.Qs.mean() / 1e6,
            'sd_Qs': cap.Qs.std(ddof=1) / 1e6,
            'E_sg': cap.sg.mean() / 1e3,
            'P_slow': (cap.t_final > V.SLOW_CUT).mean(),
            'exc_q_stag': int((cap.q_stag_max > 7e5).sum()),
            'exc_q_shldr': int((cap.q_shldr_max > 1e6).sum()),
            'exc_n': int((cap.n_max > 5.0).sum()),
        }

    designs = V.load_designs(k)
    rows = []
    print('\n================ GUIDANCE COMPARISON (MC base law vs MC '
          'Eq-2.40 law) ================', flush=True)
    for i, (name, _, _) in enumerate(designs):
        sb, sg_ = stats(b, i), stats(g, i)
        print(f"\n-- {name}", flush=True)
        for kk in ('E_Qs', 'sd_Qs', 'E_sg', 'P_slow'):
            rel = 100 * (sg_[kk] - sb[kk]) / sb[kk] if sb[kk] else float('nan')
            print(f"   {kk:<10} base {sb[kk]:10.2f}   alt {sg_[kk]:10.2f}"
                  f"   {rel:+7.1f}%", flush=True)
            rows.append({'design': name, 'quantity': kk, 'base': sb[kk],
                         'alt': sg_[kk], 'diff_pct': rel})
        for kk in ('exc_q_stag', 'exc_q_shldr', 'exc_n'):
            print(f"   {kk:<10} base {sb[kk]:4d}/{sb['captured']}"
                  f"   alt {sg_[kk]:4d}/{sg_['captured']}", flush=True)
            rows.append({'design': name, 'quantity': kk, 'base': sb[kk],
                         'alt': sg_[kk], 'diff_pct': float('nan')})
        rows.append({'design': name, 'quantity': 'captured', 'base': sb['N'],
                     'alt': sg_['N'], 'diff_pct': float('nan')})
    pd.DataFrame(rows).to_csv(G40_SUM, index=False)
    print(f"\nsaved {G40_SUM}", flush=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tt.bank_angle_sigma = make_gated_bank(eps=0.0, sin_gamma=True,
                                          cap_deg=82.0)
    V.OUT, V.SUMMARY = G40, G40_SUM + '.surrogate_ref.csv'
    print("GUIDANCE-MC: bank law patched to Eq-2.40-as-printed "
          "(sin-gamma, cap 82 deg); staging to verify_mc_g40.csv",
          flush=True)
    try:
        V.main()          # staged MC under the patched law
    finally:
        tt.bank_angle_sigma = ORIGINAL_BANK
    compare(k)


if __name__ == '__main__':
    main()
