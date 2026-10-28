"""Single-column versions of Figures 1 and 3 for the two-column PSHSJ layout.

Same data and styling as make_figs_final.py; the three raga panels are stacked
vertically so each figure fits one 3.25-inch column. Run after run_final.py.
"""
import pickle, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import zm_core as z
warnings.filterwarnings('ignore')

_HERE = Path(__file__).resolve().parent
OUT = _HERE / 'output'; OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7.5,
                 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                 'legend.fontsize': 6, 'axes.linewidth': .7, 'xtick.major.width': .7,
                 'ytick.major.width': .7, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02})

st = pickle.load(open(OUT / 'final.pkl', 'rb'))
corp, fits, perm = st['corp'], st['fits'], st['perm']
RAGAS = z.RAGAS
RC = {'Bhairav': '#1f4e79', 'Todi': '#a11d33', 'Poorvi': '#1b7837'}
SCH = {'pitch': ('#8c8c8c', 'o', 'pitch only'),
       'duration': ('#d95f0e', '^', 'duration only'),
       'pitch_duration': ('#1f4e79', 's', 'pitch x duration')}

def zm(r, q, s):
    v = (r + q) ** (-s); return v / v.sum()

def panel(ax, letter):
    ax.text(-0.13, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')

# Figure 1: unit of analysis, one row per raga
fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.0), sharex=True)
for k, (ax, raga) in enumerate(zip(axes, RAGAS)):
    notes, _ = corp[raga]
    for sch in z.SCHEMES:
        c, m, lab = SCH[sch]
        t, r, f = z.rank_frequency(z.make_tokens(notes, sch))
        ax.scatter(r, f, s=6, alpha=.55, color=c, marker=m, linewidths=0, label=f'{lab} (N={len(t)})')
        d = fits[raga][sch]
        if d['identifiable']:
            xs = np.arange(1, len(r) + 1, dtype=float)
            ax.plot(xs, zm(xs, d['q'], d['s']), '-', color=c, lw=1.0)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4, .6)
    ax.set_title(raga, color=RC[raga], fontweight='bold', pad=2)
    ax.legend(loc='lower left', frameon=False, handletextpad=.3, borderpad=.2)
    panel(ax, 'ABC'[k])
axes[-1].set_xlabel('rank $r$')
fig.supylabel('relative frequency $f(r)$', fontsize=7.5, x=0.01)
fig.tight_layout(h_pad=.5); fig.savefig(OUT / 'fig1_unit_of_analysis_column.png', dpi=300); plt.close(fig)

# Figure 3: pakad permutation (trigram), one row per raga
fig, axes = plt.subplots(3, 1, figsize=(3.3, 3.1))
for k, (ax, raga) in enumerate(zip(axes, RAGAS)):
    p = perm[raga][3]
    xhi = max(p['vocab'], p['null_ci'][1], p['null_median']) * 1.04
    ax.axvspan(p['null_ci'][0], p['null_ci'][1], color='#cfcfcf', alpha=.9, zorder=1, label='null 95% interval')
    ax.axvline(p['null_median'], color='#4d4d4d', ls='--', lw=1.0, zorder=2,
               label=f'null median = {p["null_median"]:.0f}')
    ax.axvline(p['obs_median_rank'], color=RC[raga], lw=2.0, zorder=3,
               label=f'observed = {p["obs_median_rank"]:.0f}')
    ax.set_xlim(0, xhi); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_title(f'{raga}  ($p$ = {p["p_value"]:.3f})', color=RC[raga], fontweight='bold', pad=2)
    ax.legend(loc='center right', frameon=False, handletextpad=.4, borderpad=.1, labelspacing=.2, fontsize=5.8)
    panel(ax, 'ABC'[k])
axes[-1].set_xlabel('median rank of pakad trigrams')
fig.tight_layout(h_pad=.6); fig.savefig(OUT / 'fig3_pakad_permutation_column.png', dpi=300); plt.close(fig)
print('done')
