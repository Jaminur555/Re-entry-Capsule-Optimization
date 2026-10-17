"""Step 6: deterministic vs robust NSGA-II on the surrogate (PLAN.md, decided
2026-10-08: Option A, statistical moments).

det    : thesis formulation at nominal z0 -- objectives (max eta_V, min Qs,
         max sg), constraints at point predictions; surrogate counterpart of
         capsule_opt/optimization/optimization.py.
robust : min (-eta_V, E[Qs], -E[sg], sd[Qs]); peaks as E + 2*sd <= limit
         (sd kind: 'infl' = GP-variance-inflated | 'plain' = z-dispersion
         only); P(slow) <= 0.05; geometry margins as in det.

Usage: python -m uq.optimize det|robust [POP=200] [GENS=300] [infl|plain]
                       [P_SLOW=off|<float>] [SEED=1]
SEED != 1 writes uq/data/opt_<mode>_s<seed>.csv (seed repeatability runs);
seed 1 keeps the canonical opt_<mode>.csv name.
P_SLOW: 'off' (default; diag_robust_feasible showed P_slow = 0.34-0.59 over
the whole design space -- a small budget empties the feasible set, so the
loft probability is carried as a reported quantity, not a constraint) or a
numeric budget for the P(slow) <= P_SLOW constraint.
Writes uq/data/opt_<mode>.csv (final population: designs + objectives +
constraints + all engine outputs, re-evaluated at n_z=2000 for robust).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from capsule_opt import config
from uq.engine import MomentEngine

HERE = os.path.dirname(os.path.abspath(__file__))
P_SLOW_MAX = 0.05        # diag_robust_feasible: P_slow is 0.34-0.59 over the
                         # whole space -> any small budget is infeasible;
                         # switchable to None (report-only) via CLI 'off'
N_Z_LOOP = 500           # CRN size inside NSGA-II (triangular solves cost
                         # ~0.6 s/cand at n_z=500); final front re-evaluated
                         # at n_z=2000 for reporting
N_Z_REPORT = 2000
LIMITS = {'q_stag_max': config.C_Q_STAG_MAX,
          'q_shldr_max': config.C_Q_SHLDR_MAX,
          'n_max': config.C_N_MAX}


class CapsuleSurrogate(Problem):
    """Batched surrogate problem (elementwise=False): _evaluate gets the
    whole population at once -- one moment-engine pass per generation."""

    def __init__(self, engine, mode='robust', sd_kind='infl',
                 p_slow_max=P_SLOW_MAX):
        self.eng = engine
        self.mode = mode
        self.sd_kind = sd_kind
        self.p_slow_max = p_slow_max if mode == 'robust' else None
        super().__init__(n_var=3, n_obj=4 if mode == 'robust' else 3,
                         n_ieq_constr=6 if self.p_slow_max else 5,
                         xl=np.zeros(3), xu=np.ones(3))

    def _evaluate(self, X, out, *args, **kwargs):
        eng, mode = self.eng, self.mode
        if mode == 'det':
            d = eng.det_predict(X)
            out['F'] = np.column_stack([
                -d['eta_V'], d['Qs'] / 1e7, -d['sg'] / 1e6])
            G = [(d[o] - L) / L for o, L in LIMITS.items()]
            G += [d['g_pitch_stable'], d['g_trim_on_nose']]
        else:
            m = eng.moments(X)
            out['F'] = np.column_stack([
                -m['eta_V'], m['E_Qs'] / 1e7, -m['E_sg'] / 1e6,
                m[f'sd_Qs_{self.sd_kind}'] / 1e7])
            G = [(m[f'E_{o}'] + 2.0 * m[f'sd_{o}_{self.sd_kind}'] - L) / L
                 for o, L in LIMITS.items()]
            G += [m['g_pitch_stable'], m['g_trim_on_nose']]
            if self.p_slow_max:
                G.append((m['P_slow'] - self.p_slow_max) / self.p_slow_max)
        out['G'] = np.column_stack(G)


def wide_frame(eng, X, F, G, mode, p_slow_max):
    """Designs + named F/G + full engine outputs, for figures/verification."""
    d = eng.det_predict(X) if mode == 'det' else eng.moments(X)
    df = pd.DataFrame(X, columns=['rn', 'rs', 'r_theta'])
    fcols = (['neg_eta_V', 'E_Qs_s', 'neg_E_sg_s', 'sd_Qs_s']
             if mode == 'robust' else ['neg_eta_V', 'Qs_s', 'neg_sg_s'])
    for j, c in enumerate(fcols):
        df[c] = F[:, j]
    gcols = ['g_q_stag', 'g_q_shldr', 'g_n_max', 'g_pitch', 'g_nose'] + \
            (['g_P_slow'] if mode == 'robust' and p_slow_max else [])
    for j, c in enumerate(gcols):
        df[c] = G[:, j]
    for k, v in d.items():
        df[k] = v
    return df


def main():
    if os.name == 'nt':  # keep awake while the run goes (doe.py pattern)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'robust'
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    sd_kind = sys.argv[4] if len(sys.argv) > 4 else 'infl'
    a5 = sys.argv[5] if len(sys.argv) > 5 else 'off'
    p_slow_max = None if a5 == 'off' else float(a5)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    assert mode in ('det', 'robust') and sd_kind in ('infl', 'plain')

    t0 = time.time()
    eng = MomentEngine(n_z=N_Z_LOOP)
    print(f"engine loaded {time.time() - t0:.0f} s | mode={mode} "
          f"pop={pop} gens={gens} sd={sd_kind} p_slow={p_slow_max} "
          f"n_z_loop={N_Z_LOOP} seed={seed}", flush=True)
    prob = CapsuleSurrogate(eng, mode, sd_kind, p_slow_max)
    res = minimize(prob, NSGA2(pop_size=pop),
                   get_termination('n_gen', gens), seed=seed, verbose=True)

    X, F, G = res.pop.get('X'), res.pop.get('F'), res.pop.get('G')
    eng_rep = eng if mode == 'det' else MomentEngine(n_z=N_Z_REPORT)
    df = wide_frame(eng_rep, X, F, G, mode, p_slow_max)
    feas = (G <= 0).all(axis=1)
    sfx = '' if seed == 1 else f'_s{seed}'
    out = os.path.join(HERE, f'data/opt_{mode}{sfx}.csv')
    df.to_csv(out, index=False)
    print(f"DONE {time.time() - t0:.0f} s | pop {len(df)} | feasible "
          f"{feas.mean():.1%} | saved {out}", flush=True)


if __name__ == '__main__':
    main()
