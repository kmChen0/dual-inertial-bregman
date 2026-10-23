"""Numerical routines extracted from the recorded experiment implementation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from . import regression as publication, nnls as proximal
HERE = Path(__file__).resolve().parents[1]/"results/generated/paper"
REG_METHODS = ("BPG", "AEPG", "ABPG-g", "DI-BFB", "S-DI-BFB")
VI = "V-iEPPA (q=2, heuristic)"
NNLS_METHODS = ("BPPA", VI, "DI-BPPA", "S-DI-BPPA")
proximal.STYLE[VI] = ("#CC79A7", "-.")

def plot_performance(block, cases, iteration_axis=False):
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.35), sharey=True,
                             constrained_layout=True)
    methods = REG_METHODS if block == 'regression' else NNLS_METHODS
    for panel, (value, (runs, clocks)) in enumerate(cases.items()):
        for method in methods:
            rows = runs[method]
            key = 'gap' if block == 'regression' else 'best_gap'
            gaps = np.asarray([r[key] for r in rows])/rows[0]['gap']
            counter = 'k' if iteration_axis else 'gradient_count' if block == 'regression' else 'calls'
            times = 1000*np.median(clocks[method], axis=0)
            for ax, xx in zip(axes[2*panel:2*panel+2],
                              ([r[counter] for r in rows], times)):
                if block == 'regression':
                    publication.curve(ax, xx, gaps, method)
                else:
                    proximal.plot_curve(ax, xx, gaps, method)
        axes[2*panel].set_xlabel('Outer loop iterations' if iteration_axis else
                                'Gradient evaluations' if block == 'regression' else 'Proximal calls')
        axes[2*panel+1].set_xlabel('Median update time\n(ms)')
        for index in (2*panel, 2*panel+1):
            config = f'n={value}' if block == 'regression' else rf'$\rho={value}$'
            axes[index].set_title(f'({chr(97+index)}) {config}')
        assert all(len(ax.lines) == len(methods) for ax in axes[2*panel:2*panel+2])
    axes[0].set_ylabel('Relative objective gap' if block == 'regression' else 'Relative best-point gap')
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ['V-iBPPA' if label == VI else label for label in labels]
    axes[0].legend(handles, labels, fontsize=8)
    name = f'{block}_seed1'+('_iterations' if iteration_axis else '')
    for ext in ('pdf', 'png'):
        fig.savefig(HERE/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
