"""Paper figures (Step 6): deterministic vs robust fronts + probe validation.

Publication-standard static figures (Acta Astronautica): seaborn-styled
matplotlib, boxed axes on solid white, vector PDF + 300 dpi PNG -> paper/figures.
All markers are solid circles (identity by color only: det blue vs robust
vermillion, Okabe-Ito CVD-safe); no on-figure text beyond axis labels --
reference lines and their meaning go in the paper captions. A plotly HTML
companion of the headline figure is written for interactive exploration.

Usage: python -m uq.figures        (reads uq/data/*.csv; seconds to run)
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), 'paper', 'figures')
os.makedirs(FIG, exist_ok=True)

DET, ROB, ACC = '#0072B2', '#D55E00', '#009E73'   # Okabe-Ito pair + accent
INK, MUTED = '#222222', '#666666'
SEQ = 'viridis'

sns.set_theme(style='ticks', context='paper', font_scale=1.0, rc={
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 9, 'axes.linewidth': 0.8, 'axes.edgecolor': 'black',
    'axes.grid': False,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'legend.frameon': True, 'legend.framealpha': 1.0,
    'legend.edgecolor': 'black', 'legend.fancybox': False,
    'figure.dpi': 130, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'lines.linewidth': 1.4,
})

LEG_OUT = dict(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2,
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
LEG_BELOW = dict(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2,
                 borderaxespad=0.0)


def box(ax):
    """Full box + solid white background on every panel."""
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_visible(True)
    return ax


def scat(ax, x, y, c, **kw):
    """Solid circle, no border; identity is color-only."""
    ax.scatter(x, y, s=14, c=c, marker='o', alpha=0.95, linewidths=0,
               edgecolors='none', **kw)


def load():
    d = pd.read_csv(os.path.join(HERE, 'data/opt_det_audit.csv'))
    d = d.rename(columns={c: c[4:] for c in d.columns if c.startswith('aud_')})
    d = d.loc[:, ~d.columns.duplicated()]
    r = pd.read_csv(os.path.join(HERE, 'data/opt_robust.csv'))
    scan = pd.read_csv(os.path.join(HERE, 'data/diag_robust_feasible.csv'))
    probe = pd.read_csv(os.path.join(HERE, 'data/probe_dispersion.csv'))
    return d, r, scan, probe


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'),
                    facecolor='white')
    plt.close(fig)
    print(f"  {name}.png/.pdf", flush=True)


def fig_front(d, r):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    box(ax)
    scat(ax, d.E_Qs / 1e6, d.sd_Qs_infl / 1e6, DET)
    scat(ax, r.E_Qs / 1e6, r.sd_Qs_infl / 1e6, ROB)
    floor = min(d.sd_Qs_infl.min(), r.sd_Qs_infl.min()) / 1e6
    ax.axhline(floor, c=MUTED, ls='--', lw=0.9, zorder=0)
    ax.set_ylim(bottom=floor - 0.06 * (d.sd_Qs_infl.max() / 1e6 - floor))
    ax.set_xlabel(r'$E[Q_s]$  [MJ m$^{-2}$]')
    ax.set_ylabel(r'$\sigma_{Q_s}$ (inflated)  [MJ m$^{-2}$]')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=4.5,
                              mfc=DET, mec='none', label='deterministic'),
                       Line2D([], [], ls='', marker='o', ms=4.5,
                              mfc=ROB, mec='none', label='robust')],
              loc='upper left')
    save(fig, 'fig1_front_sigma')


def fig_matched(d, r):
    """ECDFs of the per-design sigma cut under two partner-selection rules:
    the prespecified nearest-performance match and the lowest-sigma (best)
    eligible partner = the robust front's sigma-envelope at matched
    performance."""
    dd = d[['E_Qs', 'E_sg', 'sd_Qs_infl']].reset_index(names='idx_d')
    rr = r[['E_Qs', 'E_sg', 'sd_Qs_infl']].assign(k=1)
    m = dd.assign(k=1).merge(rr, on='k', suffixes=('_d', '_r'))
    ok = m[(m.E_Qs_r <= m.E_Qs_d * 1.02) & (m.E_sg_r >= m.E_sg_d * 0.98)]
    ok = ok.assign(cut=1 - ok.sd_Qs_infl_r / ok.sd_Qs_infl_d,
                   dist=np.sqrt((ok.E_Qs_r / ok.E_Qs_d - 1) ** 2 +
                                (ok.E_sg_r / ok.E_sg_d - 1) ** 2))
    best = ok.groupby('idx_d')['cut'].max()
    near = ok.loc[ok.groupby('idx_d')['dist'].idxmin(), 'cut']

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    box(ax)
    ax.plot(100 * np.sort(near), np.arange(1, len(near) + 1) / len(near),
            c=ACC, ls='--', label='nearest match (prespecified)')
    ax.plot(100 * np.sort(best), np.arange(1, len(best) + 1) / len(best),
            c=INK, label=r'best match ($\sigma$-envelope)')
    ax.axvline(0.0, c=MUTED, ls=':', lw=0.9, zorder=0)
    ax.set_xlabel(r'$\sigma_{Q_s}$ reduction at matched '
                  r'$E[Q_s],\,E[s_g]$  [%]')
    ax.set_ylabel('ECDF (deterministic designs)')
    ax.set_xlim(-100.0, 100.0)
    ax.legend(loc='lower right')
    save(fig, 'fig2_matched_sigma_cut')


def fig_violation(d, r):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, o, unit in zip(axes, ('q_stag_max', 'q_shldr_max'),
                           ('q_{stag}', 'q_{shldr}')):
        box(ax)
        ax.plot(np.sort(d[f'Pexc_{o}'].clip(1e-5, 1)),
                np.linspace(0.02, 1, len(d)), c=DET, label='deterministic')
        ax.plot(np.sort(r[f'Pexc_{o}'].clip(1e-5, 1)),
                np.linspace(0.02, 1, len(r)), c=ROB, label='robust')
        ax.set_xscale('log')
        ax.axvline(0.01, c=MUTED, ls=':', lw=0.9, zorder=0)
        ax.set_xlabel(rf'$P(\,{unit} > \mathrm{{limit}}\,)$')
        ax.set_title(rf'${unit}$', fontsize=8, pad=6)
    axes[0].set_ylabel('ECDF (designs)')
    axes[0].legend(loc='upper left')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig3_violation_ecdf')


def fig_projections(d, r):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, y, lab in zip(axes, ('eta_V', 'E_sg'),
                          (r'$\eta_V$', r'$E[s_g]$  [km]')):
        box(ax)
        yd = d[y] if y == 'eta_V' else d[y] / 1e3
        yr = r[y] if y == 'eta_V' else r[y] / 1e3
        scat(ax, d.E_Qs / 1e6, yd, DET)
        scat(ax, r.E_Qs / 1e6, yr, ROB)
        ax.set_xlabel(r'$E[Q_s]$  [MJ m$^{-2}$]')
        ax.set_ylabel(lab)
    axes[0].legend(handles=[
        Line2D([], [], ls='', marker='o', ms=4.5, mfc=DET, mec='none',
               label='deterministic'),
        Line2D([], [], ls='', marker='o', ms=4.5, mfc=ROB, mec='none',
               label='robust')],
        loc='lower right')
    fig.tight_layout(w_pad=3.0)
    save(fig, 'fig4_front_projections')


def fig_design_space(d, r, scan):
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    box(ax)
    sc = ax.scatter(5 + 55 * scan.x3, 3 + 4 * scan.x1,
                    c=scan.sd_Qs_infl / 1e6, cmap='coolwarm',
                    s=10, alpha=0.9, linewidths=0)
    ax.scatter(5 + 55 * d.r_theta, 3 + 4 * d.rn, s=17, c=DET, marker='o',
               alpha=0.95, linewidths=0.6, edgecolors='black',
               label='deterministic front')
    ax.scatter(5 + 55 * r.r_theta, 3 + 4 * r.rn, s=17, c=ROB, marker='o',
               alpha=0.95, linewidths=0.6, edgecolors='black',
               label='robust front')
    ax.set_xlabel(r'$\theta_c$  [deg]')
    ax.set_ylabel(r'$R_n$  [m]')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=4.5, mfc=DET,
                              mec='black', label='deterministic front'),
                       Line2D([], [], ls='', marker='o', ms=4.5, mfc=ROB,
                              mec='black', label='robust front')],
               loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2,
               borderaxespad=0.0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$\sigma_{Q_s}$  [MJ m$^{-2}$]', fontsize=7.5)
    save(fig, 'fig5_design_space_sigma_map')


def fig_probe(probe):
    outs = ['Qs', 'q_stag_max', 'q_shldr_max', 'n_max']
    labels = [r'$Q_s$ [MJ m$^{-2}$]', r'$q_{stag}$ [kW m$^{-2}$]',
              r'$q_{shldr}$ [kW m$^{-2}$]', r'$n_{max}$ [g]']
    scale = [1e-6, 1e-3, 1e-3, 1.0]
    dids = sorted(probe.design_id.unique())
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    box(ax)
    w = 0.19
    for k, (o, sc_) in enumerate(zip(outs, scale)):
        sds = [probe[(probe.design_id == i) & (probe.captured == 1)][o]
               .std(ddof=1) * sc_ for i in dids]
        ax.bar(np.arange(len(dids)) + (k - 1.5) * w, sds, w,
               color=plt.cm.coolwarm([0.12, 0.38, 0.62, 0.88][k]),
               label=labels[k], edgecolor='white', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(dids)))
    ax.set_xticklabels([f'D{i}' for i in dids])
    ax.set_ylabel(r'true $\sigma_z$ (log scale)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4,
              handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, 'fig6_probe_true_sigma')


def fig_front_plotly(d, r):
    """Interactive HTML companion of the headline figure."""
    import plotly.express as px
    import plotly.graph_objects as go
    dd = d.assign(front='deterministic', x=d.E_Qs / 1e6,
                  y=d.sd_Qs_infl / 1e6)
    rr = r.assign(front='robust', x=r.E_Qs / 1e6, y=r.sd_Qs_infl / 1e6)
    df = pd.concat([dd, rr])[['front', 'x', 'y', 'rn', 'rs', 'r_theta',
                              'P_slow', 'E_sg']]
    fig = px.scatter(df, x='x', y='y', color='front', symbol=None,
                     color_discrete_map={'deterministic': DET,
                                         'robust': ROB},
                     hover_data=['rn', 'rs', 'r_theta', 'P_slow', 'E_sg'],
                     labels={'x': 'E[Qs] [MJ/m2]',
                             'y': 'sigma_Qs inflated [MJ/m2]'})
    floor = float(df.y.min())
    fig.add_hline(y=floor, line_dash='dash', line_color='#666666',
                  annotation_text=f'irreducible floor {floor:.1f}',
                  annotation_position='bottom right')
    fig.update_layout(template='plotly_white', font_family='Times New Roman',
                      title='Deterministic vs robust front')
    out = os.path.join(FIG, 'fig1_interactive.html')
    fig.write_html(out)
    print(f"  fig1_interactive.html", flush=True)


def main():
    d, r, scan, probe = load()
    print(f"figures -> {FIG}", flush=True)
    fig_front(d, r)
    fig_matched(d, r)
    fig_violation(d, r)
    fig_projections(d, r)
    fig_design_space(d, r, scan)
    fig_probe(probe)
    fig_front_plotly(d, r)
    print("done.", flush=True)


if __name__ == '__main__':
    main()
