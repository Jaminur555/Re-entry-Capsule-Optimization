"""Thesis-derived publication figures (revision Phase C, 2026-10-12).

Builds the paper's remaining figures from thesis material and repo data,
in the same house style as uq.figures (import the theme from there; never
edit the paper/ copies).  Figures:

  fig7_silhouettes_knees       meridional outlines: det/robust knees +
                               robust matched + Apollo (Sec. 5, tab:knees)
  fig_validation_aero          Apollo CD/CL/Cm/L/D at M=10 vs digitized
                               D&M LIM + wind tunnel (Sec. 2.4, V2)
  figS2_validation_aero_M3     same at M=3 (supplementary)
  fig_validation_traj          Apollo entry profiles vs digitized
                               Fig. 7.17: altitude, L/D, heat rate,
                               load factor (Sec. 2.4, V4b)
  fig_validation_heating       Zoby-Sullivan effective-radius chart +
                               shoulder-ratio parity (Sec. 2.4, V3)

Usage: python -m uq.figures_thesis    (one trajectory run, ~1 min)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capsule_opt import config
from capsule_opt.aerodynamics.solver import HypersonicAeroSolver
from capsule_opt.atmosphere import atmosphere
from capsule_opt.heating.heating import (effective_nose_radius,
                                         shoulder_heat_flux)
from capsule_opt.optimization.metrics import heat_load, load_factor
from capsule_opt.properties import get_capsule_properties
from capsule_opt.trajectory.trajectory import bank_angle_sigma
from uq.figures import DET, ROB, INK, MUTED, FIG, LEG_BELOW, box

import validation.apollo_case as apollo

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.join(os.path.dirname(HERE), 'paper', 'supplementary')
REFD = os.path.join(os.path.dirname(HERE), 'validation', 'reference_data')

# reference-data markers: digitized book = black open circle,
# wind tunnel = filled vermillion circle
REF_KW = dict(marker='o', ms=3.2, mfc='none', mec=INK, mew=0.8, ls='none')
WT_KW = dict(marker='o', ms=3.4, mfc=ROB, mec=ROB, mew=0.8, ls='none')


# --------------------------------------------------------------- helpers
def outline(rn, rs, r_theta):
    """Meridional (x, z) contour of one capsule (normalized inputs)."""
    p = get_capsule_properties(rn, rs, r_theta)
    return p['X'], p['Y']


def apollo_outline():
    """Apollo CM reference contour (book geometry), temporarily swapping
    the fixed Rm/Lc, as in the thesis silhouette figure."""
    saved = (config.Rm_fixed, config.Lc_fixed)
    config.Rm_fixed, config.Lc_fixed = 1.956, 2.662
    try:
        rna = (4.694 - config.RN_MIN) / (config.RN_MAX - config.RN_MIN)
        rsa = (0.196 - config.RS_MIN) / (config.RS_MAX - config.RS_MIN)
        rta = (33.0 - config.THETA_C_MIN) / (config.THETA_C_MAX -
                                             config.THETA_C_MIN)
        return outline(rna, rsa, rta)
    finally:
        config.Rm_fixed, config.Lc_fixed = saved


def _save(fig, name, folder=FIG):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name), bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name} (.png/.pdf)', flush=True)


# --------------------------------------------------------------- figures
def silhouette_knees():
    """fig7: Dirkx-style overlaid meridional outlines of the knee designs."""
    knee = pd.read_csv(os.path.join(HERE, 'data', 'knee_designs.csv')
                       ).set_index('mode')
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    box(ax)

    # Knee outlines at the STUDY geometry (Rm=2, Lc=2).  Importing
    # validation.apollo_case (needed by the validation figures in this
    # module) fixes config at Apollo values for the whole process, so the
    # study outlines must swap back explicitly.
    saved = (config.Rm_fixed, config.Lc_fixed)
    config.Rm_fixed, config.Lc_fixed = 2.0, 2.0
    try:
        for mode, color, lab in (
                ('det', DET, 'deterministic knee'),
                ('robust', ROB, 'robust knee'),
                ('robust_m', MUTED, 'robust matched')):
            r = knee.loc[mode]
            X, Z = outline(r.rn, r.rs, r.r_theta)
            ax.plot(X, Z, color=color, lw=1.6, label=lab)
    finally:
        config.Rm_fixed, config.Lc_fixed = saved

    Xa, Za = apollo_outline()
    ax.plot(Xa, Za, color=INK, lw=1.1, ls='--', label='Apollo CM (reference)')

    ax.set_aspect('equal')
    ax.set_xlabel(r'axial coordinate $x$ [m]')
    ax.set_ylabel(r'radial coordinate $z$ [m]')
    ax.legend(**{**LEG_BELOW, 'ncol': 2})
    _save(fig, 'fig7_silhouettes_knees')


def validation_aero(mach=10.0, name='fig_validation_aero', folder=FIG):
    """Apollo force coefficients at one Mach vs digitized D&M LIM + WT."""
    ref = pd.read_csv(os.path.join(REFD, 'apollo_flight_wind_tunnel_data.csv'))
    ref = ref[ref.mach == mach]

    props = apollo.apollo_props()
    solver = HypersonicAeroSolver()
    scale = apollo.cm_ref_scale()
    alphas = np.linspace(0.0, 30.0, 61)
    res = [solver.coefficients(props, mach, a) for a in alphas]
    ours = {'CD': np.array([r['CD'] for r in res]),
            'CL': np.array([r['CL'] for r in res]),
            'Cm': -np.array([r['Cm'] for r in res]) * scale,
            'L/D': np.array([r['L/D'] for r in res])}

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), constrained_layout=True)
    for ax, c in zip(axes.ravel(), ('CD', 'CL', 'Cm', 'L/D')):
        box(ax)
        d = ref[ref.coeff == c]
        lim = d[d.source == 'lim']
        wt = d[d.source != 'lim']
        ax.plot(alphas, ours[c], color=DET, lw=1.5)
        ax.plot(-lim.alpha_deg, lim.value, **REF_KW)
        if len(wt):
            ax.plot(-wt.alpha_deg, wt.value, **WT_KW)
        ax.set_xlabel(r'$\alpha$ [deg]')
        ax.set_ylabel({'Cm': r'$C_m$', 'L/D': r'$L/D$'}.get(c, f'${c}$'))
        if c in ('CL', 'L/D'):
            ax.set_yscale('symlog', linthresh=0.05)
    axes[0, 0].plot([], [], color=DET, lw=1.5, label='present model')
    axes[0, 0].plot([], [], **{**REF_KW, 'label': 'D\\&M LIM (digitized)'})
    axes[0, 0].plot([], [], **{**WT_KW, 'label': 'wind tunnel'})
    axes[0, 0].legend(loc='lower left', handletextpad=0.3)
    _save(fig, name, folder)


def validation_traj():
    """Apollo entry profiles vs digitized Fig. 7.17 (altitude, L/D, q, n)."""
    d = pd.read_csv(os.path.join(REFD, 'apollo_entry_reference_dm2017.csv'))

    def dig(profile, indep='mach'):
        s = d[(d.profile == profile) & (d.indep_name == indep) &
              (d.aero_source == 'lim')]
        x = s.indep_value.to_numpy(float)
        v = s.value.to_numpy(float)
        o = np.argsort(x)
        return x[o], v[o]

    traj = apollo.run_entry()
    iCD, iCL, _iCM, _atr = apollo.apollo_database()
    props = apollo.apollo_props()
    m_kg = apollo.ENTRY_ICS['mass_kg']
    A_ref = props['A_ref']

    M_all = np.asarray(traj['Mach'], float)
    i_pk = int(np.argmax(M_all))       # profiles live on the descending branch
    sl = slice(i_pk, None)
    M = M_all[sl]
    o = np.argsort(M)

    at = np.clip(np.asarray(traj['alpha_trim'], float)[sl],
                 config.aoa_nodes[0], config.aoa_nodes[-1])
    Mc = np.clip(M, config.mach_nodes[0], config.mach_nodes[-1])
    CD = np.array([float(iCD(np.array([[Mc[k], at[k]]]))[0])
                   for k in range(len(M))])
    CL = np.array([float(iCL(np.array([[Mc[k], at[k]]]))[0])
                   for k in range(len(M))])
    heat = heat_load(traj, apollo.APOLLO_RN_NORM, apollo.APOLLO_RS_NORM,
                     apollo.APOLLO_RTHETA_NORM)
    nres = load_factor(traj, apollo.APOLLO_RN_NORM, apollo.APOLLO_RS_NORM,
                       apollo.APOLLO_RTHETA_NORM, m_kg,
                       shape_props=props, interp_CD=iCD)

    panels = [
        ('altitude', np.asarray(traj['t'], float),
         np.asarray(traj['h'], float) / 1e3,
         ('t [s]', 'h [km]'), *dig('altitude_km', 'time_s')),
        ('L/D', M[o], (CL / CD)[o], ('Mach number', 'L/D'),
         *dig('lift_to_drag')),
        ('heat rate', M[o], np.asarray(heat['q_stag_arr'])[sl][o] / 1e3,
         ('Mach number', r'$\dot{q}_{\mathrm{stag}}$ [kW/m$^2$]'),
         *(lambda x, v: (x, v))(*dig('heat_rate_w_m2'))),
        ('load factor', M[o], np.asarray(nres['n_arr'])[sl][o],
         ('Mach number', r'$n$ [g]'), *dig('load_factor_g')),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), constrained_layout=True)
    for ax, (_lab, xo, yo, (xl, yl), xr, vr) in zip(axes.ravel(), panels):
        box(ax)
        if _lab == 'L/D' and np.mean(vr) < 0:
            vr = -vr                    # D&M negative-alpha sign convention
        ax.plot(xo, yo, color=DET, lw=1.5)
        sel = (xr >= (xo.min() if _lab != 'altitude' else xr.min())) & \
              (xr <= xo.max())
        ax.plot(xr[sel], vr[sel], **REF_KW)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0, 0].plot([], [], color=DET, lw=1.5, label='present model')
    axes[0, 0].plot([], [], **{**REF_KW, 'label': 'D\\&M LIM (digitized)'})
    axes[0, 0].legend(loc='upper right', handletextpad=0.3)
    _save(fig, 'fig_validation_traj')


def validation_heating():
    """Zoby-Sullivan effective-radius chart + shoulder-ratio parity."""
    zs = pd.read_csv(os.path.join(REFD, 'zoby_sullivan_chart_digitized.csv'))
    sh = pd.read_csv(os.path.join(REFD, 'shoulder_ratio_reference.csv'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                 constrained_layout=True)
    box(a1)
    keys = sorted(zs.rs_over_rm.unique())
    for k in keys:
        g = zs[zs.rs_over_rm == k].sort_values('rm_over_rn')
        x = np.linspace(0.0, 1.0, 101)         # our in-code chart, Rn sweep
        Rn = config.Rm_fixed / np.maximum(x, 1e-6)
        Rs = k * config.Rm_fixed
        ours = np.array([config.Rm_fixed /
                         effective_nose_radius(rn, Rs) for rn in Rn])
        a1.plot(x, ours, color=DET, lw=1.3)
        a1.plot(g.rm_over_rn, g.rm_over_reff, **REF_KW)
    a1.set_xlabel(r'$R_m/R_n$')
    a1.set_ylabel(r'$R_m/R_{\mathrm{eff}}$')
    a1.plot([], [], color=DET, lw=1.3, label='present chart')
    a1.plot([], [], **{**REF_KW, 'label': 'digitized chart'})
    a1.legend(loc='upper left', handletextpad=0.3)

    box(a2)
    theta_sp1 = get_capsule_properties(apollo.APOLLO_RN_NORM,
                                       apollo.APOLLO_RS_NORM,
                                       apollo.APOLLO_RTHETA_NORM)['theta_sp1']
    model = np.array([shoulder_heat_flux(1.0, r.mach, r.alpha_deg,
                                         theta_sp1, r.rs_over_rm *
                                         config.Rm_fixed)
                      for r in sh.itertuples()])
    a2.plot([0, sh.reference_ratio.max() * 1.1],
            [0, sh.reference_ratio.max() * 1.1], color=INK, lw=0.8)
    for s, m in (('marvin_sinclair', 'o'), ('jones', 's'), ('wadham', '^')):
        msk = sh.source == s
        a2.plot(sh.reference_ratio[msk], model[msk], marker=m, ms=3.5,
                mfc=DET, mec=DET, ls='none',
                label=s.replace('_', ' ') if msk.any() else None)
    lim = sh.reference_ratio.max() * 1.1
    a2.plot([0, lim], [0, lim * 1.1], color=INK, lw=0.8)
    a2.set_xlabel(r'reference $q_{\mathrm{shldr}}/q_{\mathrm{stag}}$')
    a2.set_ylabel(r'present model ratio')
    a2.legend(loc='upper left', handletextpad=0.3)
    _save(fig, 'fig_validation_heating')


def thesis_overlay():
    """fig8: thesis 25k-eval deterministic front vs the present det front.

    Both are nominal-point formulations of the same three-objective
    problem; the thesis front used the modified-Newtonian aero and
    simulator evaluation, the present front the validated local-inclination
    chain on the surrogate at z0.
    """
    import shutil
    src = r'F:\My documents\Thesis Extension\Thesis\Capsule_optimization' \
          r'\results_outputs\pareto_designs.csv'
    dst = os.path.join(os.path.dirname(HERE), 'paper', 'data',
                       'thesis_front_25keval.csv')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

    t = pd.read_csv(dst)
    d = pd.read_csv(os.path.join(HERE, 'data', 'opt_det.csv'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                 constrained_layout=True)
    for ax in (a1, a2):
        box(ax)
        ax.set_xlabel(r'$Q_s$ [MJ/m$^2$]')
    a1.scatter(d.Qs / 1e6, d.eta_V, s=13, c='#0072B2', edgecolors='none')
    a1.scatter(t.Qs_MJ_m2, t.eta_V, s=15, c='#888888', edgecolors='none')
    a1.set_ylabel(r'$\eta_V$ [-]')
    a2.scatter(d.Qs / 1e6, d.sg / 1e3, s=13, c='#0072B2', edgecolors='none')
    a2.scatter(t.Qs_MJ_m2, t.Sg_km / 1e3, s=15, c='#888888', edgecolors='none')
    a2.set_ylabel(r'$sg$ [10$^3$ km]')
    a1.scatter([], [], s=13, c='#0072B2', label='present det. front')
    a1.scatter([], [], s=15, c='#888888',
               label='thesis front (MN aero, 25k eval.)')
    a1.legend(loc='upper right', handletextpad=0.2, borderpad=0.4)
    _save(fig, 'fig8_thesis_front_overlay')


def main():
    os.makedirs(FIG, exist_ok=True)
    silhouette_knees()
    validation_aero(10.0, 'fig_validation_aero', FIG)
    validation_aero(3.0, 'figS2_validation_aero_M3', SUP)
    validation_traj()
    validation_heating()
    # thesis_overlay(): NOT in the manuscript (user decision 2026-10-14 --
    # the thesis is not treated as a reference); function retained above in
    # case the thesis is ever formally cited and the comparison reinstated.


if __name__ == '__main__':
    main()
