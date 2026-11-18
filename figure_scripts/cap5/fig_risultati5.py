"""Differenza di Macro-F1 rispetto al modello di riferimento, DVS-Lip e THU-EACT-50-CHL (sezione 5.4).

Simbolo pieno: differenza significativa nel confronto appaiato con il riferimento (McNemar esatto, p < 0,05).
Dati da risultati5.py (validazione di sviluppo).
"""
import numpy as np
from matplotlib.lines import Line2D
from common5 import *
from risultati5 import RUN, metrics, paired

ANTI = "#2a78d6"
ROWS = [('lineare', 'IRM lineare'), ('simmetrico', 'IRM simmetrico'), ('antisimmetrico', 'IRM antisimmetrico'),
        ('instradato', 'IRM con instradamento'), (None, None),
        ('vuoto', 'sito vuoto'), ('mlp', 'MLP di pari parametri'), ('punti', 'relazione a punti'),
        ('gate', 'gate di contenuto')]
SETS = [('dvslip', 'DVS-Lip'), ('thu_eact_50_chl', 'THU-EACT-50-CHL')]

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
y = np.arange(len(ROWS))[::-1]
for ax, (ds, name) in zip(axes, SETS):
    ref = metrics(ds, 'rif')['f1']
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.axhline(y[4], color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for yi, (key, lab) in zip(y, ROWS):
        if key is None:
            continue
        d = metrics(ds, key)['f1'] - ref
        sig = paired(ds, 'rif', key)['p'] < 0.05
        col = ANTI if key == 'antisimmetrico' else INK2
        ax.plot([0, d], [yi, yi], color=GRID if key != 'antisimmetrico' else tint(ANTI, 0.45), lw=2.4,
                solid_capstyle='butt', zorder=2)
        ax.plot(d, yi, 'o', ms=6.5, color=col if sig else 'white', markeredgecolor=col, markeredgewidth=1.3,
                zorder=3)
        ax.text(d + (0.75 if d >= 0 else -0.75), yi, (r"$+%s$" if d >= 0 else r"$%s$") % num(d, 1),
                ha='left' if d >= 0 else 'right', va='center', fontsize=7, color=INK)
    ax.set_xlim(-6.8, 10.8)
    ax.xaxis.set_major_formatter(comma(0))
    ax.set_xlabel(r"differenza di Macro-F1 [punti]", fontsize=7.5)
    ax.set_title(f"{name}  (riferimento ${num(ref, 1)}$)", fontsize=8.5)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks([yi for yi, (k, _) in zip(y, ROWS) if k])
axes[0].set_yticklabels([lab for k, lab in ROWS if k], fontsize=7.5)
handles = [Line2D([], [], marker='o', ls='', ms=6, color=INK2, label=r'differenza significativa ($p<0{,}05$)'),
           Line2D([], [], marker='o', ls='', ms=6, color='white', markeredgecolor=INK2, markeredgewidth=1.3,
                  label='differenza non significativa')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.55, -0.17), ncol=2, fontsize=7.2)
save(fig, 'cap5_risultati')
