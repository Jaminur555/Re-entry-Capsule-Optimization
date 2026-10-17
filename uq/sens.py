"""Reviewer-requested sensitivity studies (2026-10-12, Phase A of revision).

Two questions, both answerable WITHOUT new simulator runs by re-evaluating
the two saved fronts through the moment engine under alternative
uncertainty models:

1. Do the headline comparison statistics survive plausible changes in the
   uncertainty model?  Variants: sigma_k = 0.03 / 0.07 (aero model-form),
   rho(k_CD, k_CL) = 0.8 (correlated scales -- the fitted Mach-wise scale
   factors of the validation moved together), sigma_gamma0 = 0.25 / 0.75 deg
   (entry-state dispersion).

2. Do the matched-pair statistics depend on the matching window?
   Windows w = 1 / 2 / 3 / 5 % at the baseline uncertainty model.

Method: for each variant, rebuild the (n_z=2000, seed 42) LHS with the same
underlying uniforms (so gamma0/V0 draws are shared with the baseline
wherever possible), override engine.Z, and recompute moments for BOTH
fronts.  Every statistic is then recomputed exactly as in uq/tables.py
table4 / the audit.  The 'base' variant must reproduce the published
numbers (5,371 pairs / 98.1% / 27.5% / 35.0%, sigma floor 16.4, det Pexc
shldr max 10.0%) -- it is the correctness anchor.

Output: paper/tables/table6_sensitivity.csv (long format) + console table.

Usage:  python -m uq.sens          (~30-60 min, no simulator runs)
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from uq.engine import MomentEngine, LIMITS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
TAB = os.path.join(os.path.dirname(HERE), 'paper', 'tables')

N_Z = 2000
Z_SEED = 42
W_BASE = 0.02          # baseline matching window (E_Qs +2%, E_sg -2%)
PEAKS = ('q_stag_max', 'q_shldr_max', 'n_max')


# ------------------------------------------------------------------ sampling
def build_Z(sigma_g=0.5, sigma_k=0.05, rho=0.0, n=N_Z, seed=Z_SEED):
    """Variant z-sample on the baseline LHS structure.

    gamma0 / V0 keep the baseline LHS uniforms (same quantiles for any
    sigma).  For rho = 0 the k's likewise come from the LHS uniforms; for
    correlated k's a Gaussian copula (plain MC, seed 43) replaces the two
    LHS columns -- marginals exact, correlation induced, stratification on
    those two columns only is lost (negligible at n = 2000 for moments).
    """
    u = stats.qmc.LatinHypercube(d=4, seed=seed).random(n)
    z = np.empty((n, 4))
    z[:, 0] = stats.norm(-2.0, sigma_g).ppf(u[:, 0])
    z[:, 1] = stats.norm(7830.0, 50.0).ppf(u[:, 1])
    if rho == 0.0:
        z[:, 2] = stats.norm(1.0, sigma_k).ppf(u[:, 2])
        z[:, 3] = stats.norm(1.0, sigma_k).ppf(u[:, 3])
    else:
        # correlated k_CD, k_CL
        c = np.array([[1.0, rho], [rho, 1.0]])
        v = (np.random.default_rng(43).standard_normal((n, 2)) @
             np.linalg.cholesky(c).T)
        z[:, 2] = stats.norm(1.0, sigma_k).ppf(stats.norm.cdf(v[:, 0]))
        z[:, 3] = stats.norm(1.0, sigma_k).ppf(stats.norm.cdf(v[:, 1]))
    return z


VARIANTS = [
    ('base',  dict()),
    ('k03',   dict(sigma_k=0.03)),
    ('k07',   dict(sigma_k=0.07)),
    ('kcorr', dict(rho=0.8)),
    ('g025',  dict(sigma_g=0.25)),
    ('g075',  dict(sigma_g=0.75)),
]


# ------------------------------------------------------------------ stats
def front_stats(mD, mR, w=W_BASE):
    """Audit + matched-pair statistics for one (det, robust) moment pair.

    mD/mR are MomentEngine.moments() outputs for the two fronts, evaluated
    under the SAME variant sample (the comparison lens).
    """
    sdD, sdR = mD['sd_Qs_infl'], mR['sd_Qs_infl']
    cvD = sdD / mD['E_Qs']

    def robfeas(m):
        g = [m[f'E_{o}'] / LIMITS[o] - 1.0 + 2.0 * m[f'sd_{o}_infl'] / LIMITS[o]
             for o in PEAKS]
        return np.all(np.vstack(g) <= 0.0, axis=0) & \
            (m['g_pitch_stable'] <= 0.0) & (m['g_trim_on_nose'] <= 0.0)

    # matched pairs (uq/tables.py table4 semantics)
    m = pd.DataFrame({'E_Qs_d': mD['E_Qs'], 'E_sg_d': mD['E_sg'], 'sd_d': sdD,
                      'i': np.arange(len(sdD))}).assign(k=1).merge(
        pd.DataFrame({'E_Qs_r': mR['E_Qs'], 'E_sg_r': mR['E_sg'],
                      'sd_r': sdR}).assign(k=1), on='k')
    ok = m[(m.E_Qs_r <= m.E_Qs_d * (1 + w)) & (m.E_sg_r >= m.E_sg_d * (1 - w))]
    ok = ok.assign(cut=1 - ok.sd_r / ok.sd_d,
                   dist=np.sqrt((ok.E_Qs_r / ok.E_Qs_d - 1) ** 2 +
                                (ok.E_sg_r / ok.E_sg_d - 1) ** 2))
    best = ok.groupby('i')['cut'].max()
    near = ok.loc[ok.groupby('i')['dist'].idxmin()]

    return {
        'det_sdQs_median [MJ/m2]': np.median(sdD),
        'det_CV_median [-]': np.median(cvD),
        'det_Pexc_shldr_max [-]': mD['Pexc_q_shldr_max'].max(),
        'det_frac_Pexc_shldr_gt1pct [-]': (mD['Pexc_q_shldr_max'] > 0.01).mean(),
        'det_frac_Pexc_stag_gt1pct [-]': (mD['Pexc_q_stag_max'] > 0.01).mean(),
        'floor_sdQs_robust [MJ/m2]': sdR.min(),
        'det_robust_feasible_frac [-]': robfeas(mD).mean(),
        'matched_pairs [-]': len(ok),
        'pair_frac_favor_robust [-]': (ok.cut > 0).mean(),
        'pair_median_cut [-]': ok.cut.median(),
        'design_matched_n [-]': int(ok.i.nunique()),
        'design_frac_beaten [-]': (best > 0).mean(),
        'design_median_cut [-]': best.median(),
        'nearest_frac_beaten [-]': (near.cut > 0).mean(),
        'nearest_median_cut [-]': near.cut.median(),
    }


def main():
    det = pd.read_csv(os.path.join(DATA, 'opt_det.csv'))
    rob = pd.read_csv(os.path.join(DATA, 'opt_robust.csv'))
    Xd = det[['rn', 'rs', 'r_theta']].to_numpy()
    Xr = rob[['rn', 'rs', 'r_theta']].to_numpy()

    eng = MomentEngine(n_z=N_Z, seed=Z_SEED)
    baseZ = eng.Z.copy()

    rows = []
    for name, kw in VARIANTS:
        Z = build_Z(**kw)
        if name == 'base':
            assert np.allclose(Z, baseZ), 'base variant must equal engine Z'
        eng.Z = Z
        print(f"[{name}] moments det(n={len(Xd)}) + rob(n={len(Xr)}) ...",
              flush=True)
        st = front_stats(eng.moments(Xd), eng.moments(Xr))
        for k, v in st.items():
            rows.append({'variant': name, 'statistic': k,
                         'value': v / 1e6 if 'MJ/m2' in k else v})
        print(f"[{name}] floor={st['floor_sdQs_robust [MJ/m2]']/1e6:.1f} "
              f"detSdMed={st['det_sdQs_median [MJ/m2]']/1e6:.1f} "
              f"pairs={st['matched_pairs [-]']} "
              f"favor={st['pair_frac_favor_robust [-]']:.3f} "
              f"designcut={st['design_median_cut [-]']:.3f}", flush=True)

    # matching-window sensitivity at baseline uncertainty
    eng.Z = baseZ
    print("[windows] base moments cached per front ...", flush=True)
    mD, mR = eng.moments(Xd), eng.moments(Xr)
    for w in (0.01, 0.02, 0.03, 0.05):
        st = front_stats(mD, mR, w=w)
        for k, v in st.items():
            if 'match' in k or 'pair' in k or 'design' in k or 'favor' in k \
                    or 'cut' in k:
                rows.append({'variant': f'win{int(w*100)}', 'statistic': k,
                             'value': v})
        print(f"[win{int(w*100)}] pairs={st['matched_pairs [-]']:>6,.0f} "
              f"favor={st['pair_frac_favor_robust [-]']:.3f} "
              f"paircut={st['pair_median_cut [-]']:.3f} "
              f"designcut={st['design_median_cut [-]']:.3f}", flush=True)

    os.makedirs(TAB, exist_ok=True)
    out = os.path.join(TAB, 'table6_sensitivity.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwritten: {out}", flush=True)


if __name__ == '__main__':
    main()
