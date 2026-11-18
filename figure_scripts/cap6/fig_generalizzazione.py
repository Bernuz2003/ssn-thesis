"""Generalizzazione e voto fra varianti (sezione 6.2).

(a) accuratezza di addestramento (media delle ultime 16 epoche) e di validazione (checkpoint selezionato) del
    modello di riferimento e dell'IRM antisimmetrico a 0,5, 1 e 2 milioni di parametri;
(b) quota dei campioni di validazione riconosciuti da 0, 1, ..., 5 delle cinque varianti compatte del voto
    (stesse varianti degli insegnanti della distillazione). Dati da risultati6.py.
"""
import numpy as np
from matplotlib.lines import Line2D
from common6 import *
from risultati6 import metrics, train_u16, voto

MODELS = [('rif', 'riferimento'), ('antisimmetrico', 'IRM'), ('irm_1m', 'IRM, 1 M'), ('irm_2m', 'IRM, 2 M')]
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1]))

ax = axes[0]
labels, y = [], 0
ys = []
for ds in ('dvslip', 'thu_eact_50_chl'):
    for key, lab in MODELS:
        tr, va = train_u16(ds, key), metrics(ds, key)['acc']
        col = DSCOL[ds]
        ax.plot([va, tr], [y, y], color=tint(col, 0.35), lw=2.2, solid_capstyle='butt', zorder=2)
        ax.plot(tr, y, 'o', ms=6, color='white', markeredgecolor=col, markeredgewidth=1.3, zorder=3)
        ax.plot(va, y, 'o', ms=6, color=col, zorder=3)
        labels.append(lab); ys.append(y); y -= 1
    y -= 0.7
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7.3)
ax.set_xlim(52, 103)
ax.set_xlabel(r"accuratezza [\%]", fontsize=7.5)
ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.text(53, ys[0] + 0.55, DSNAME['dvslip'], fontsize=7.6, color=INK, va='bottom')
ax.text(53, ys[4] + 0.55, DSNAME['thu_eact_50_chl'], fontsize=7.6, color=INK, va='bottom')
ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.2)
ax.set_title(r"\textbf{(a)} addestramento e validazione", loc='left', fontsize=8.2)
h = [Line2D([], [], marker='o', ls='', ms=5.5, color='white', markeredgecolor=INK2, markeredgewidth=1.2,
            label='addestramento'),
     Line2D([], [], marker='o', ls='', ms=5.5, color=INK2, label='validazione')]
ax.legend(handles=h, loc='lower left', fontsize=6.8, handletextpad=0.2, borderaxespad=0.1)

ax = axes[1]
k = np.arange(6)
for j, ds in enumerate(('dvslip', 'thu_eact_50_chl')):
    n = voto(ds)['n_corretti']
    share = 100 * n / n.sum()
    ax.bar(k + (j - 0.5) * 0.38, share, width=0.36, color=DSCOL[ds], label=DSNAME[ds], zorder=2)
ax.set_xticks(k)
ax.set_xlabel("varianti che riconoscono il campione", fontsize=7.5)
ax.set_ylabel(r"campioni [\%]", fontsize=7.5)
ax.yaxis.set_major_formatter(comma(0))
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.legend(loc='upper left', fontsize=6.8)
ax.set_title(r"\textbf{(b)} accordo fra cinque varianti", loc='left', fontsize=8.2)
save(fig, 'cap6_generalizzazione')
