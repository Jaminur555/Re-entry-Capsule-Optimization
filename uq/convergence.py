"""NSGA-II convergence evidence (Step-6 hardening, 2026-10-10).

Parses the pymoo verbose blocks from the optimization tee logs and plots
per-generation non-dominated count (n_nds, saturating at the population
size = front filled) and design-space movement (eps, pymoo's convergence
proxy, decaying to a recombination noise floor).

Runs: robust seed 1 (opt_robust_log.txt) + det seeds 2/3 (repeatability
runs; the det seed-1 run predates logging). Style: uq.figures (imported
for the shared theme/palette).

Usage: python -m uq.convergence
  -> paper/supplementary/figS1_nsga2_convergence.{png,pdf} + console summary
"""

import os
import re

import numpy as np

from uq.figures import DET, ROB, box, plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUP = os.path.join(os.path.dirname(HERE), 'paper', 'supplementary')
LOGS = [('robust, seed 1', ROB, '-', 'opt_robust_log.txt'),
        ('deterministic, seed 2', DET, '-', 'opt_det_s2_log.txt'),
        ('deterministic, seed 3', DET, '--', 'opt_det_s3_log.txt')]
ROW = re.compile(r'^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*\S+\s*\|'
                 r'\s*\S+\s*\|\s*(\S+)')


def parse(path):
    gen, nds, eps = [], [], []
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if not m:
                continue
            g, _, n, e = m.groups()
            gen.append(int(g))
            nds.append(int(n))
            if e != '-':
                eps.append((int(g), float(e)))
    return np.array(gen), np.array(nds), eps


def main():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    for label, c, ls, fn in LOGS:
        p = os.path.join(HERE, 'logs', fn)
        if not os.path.exists(p):
            print(f"  skip {fn} (not found)", flush=True)
            continue
        gen, nds, eps = parse(p)
        a1.plot(gen, nds, c=c, ls=ls, lw=1.2, label=label)
        ge = np.array([e[0] for e in eps])
        ev = np.array([e[1] for e in eps])
        a2.plot(ge, ev, c=c, ls=ls, lw=1.2)
        pop = nds.max()
        full = int(gen[np.argmax(nds == pop)]) if (nds == pop).any() else -1
        tail = [e for e in eps if e[0] > gen.max() - 50]
        print(f"  {label}: {len(gen)} gens | n_nds -> {nds[-1]}/{pop} "
              f"(first full at gen {full}) | median eps last-50 "
              f"{np.median([t[1] for t in tail]):.2e}", flush=True)

    for a in (a1, a2):
        box(a)
    a1.set_ylabel('non-dominated in population')
    a1.legend(loc='lower right', frameon=True)
    a2.set_yscale('log')
    a2.set_ylabel(r'design-space movement $\varepsilon$')
    a2.set_xlabel('generation')
    fig.tight_layout()
    os.makedirs(SUP, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(SUP, f'figS1_nsga2_convergence.{ext}'),
                    facecolor='white')
    plt.close(fig)
    print(f"saved paper/supplementary/figS1_nsga2_convergence.png/.pdf",
          flush=True)


if __name__ == '__main__':
    main()
