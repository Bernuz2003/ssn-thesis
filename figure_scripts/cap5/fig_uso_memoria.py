"""Che cosa usa la soluzione appresa (sezione 5.5): due figure.

cap5_bande_usate: variazione di accuratezza sulla validazione di sviluppo quando si azzerano W_k e U_k di una sola
    banda nel secondo patch embedding (intervento sul checkpoint, sola inferenza). Fonte:
    artifacts/reports/rt_fbc_comparison/ablations.json, chiavi '<dataset>|ri', condizioni trace_k_off.
cap5_ordine: accuratezza sul test ufficiale del modello di riferimento e dell'IRM antisimmetrico con i passi
    nell'ordine originale, invertito e rimescolato (media di tre permutazioni). Fonte:
    artifacts/reports/c0_ri_diagnostics/csv/E2.csv.

Nota: su DVS-Lip entrambe le fonti usano per ora il checkpoint IRM 20261026_112001; da aggiornare con le
valutazioni sul checkpoint di riferimento (20261112_082513) quando disponibili.
"""
import csv, json
import numpy as np
from common5 import *

REP = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports'
SETS = [('dvslip', 'DVS-Lip', ['1--2', '3--4', '5--7', '8--15']),
        ('thu_eact_50_chl', 'THU-EACT-50-CHL', ['1--3', '4--6', '7--11', '12--23'])]
ANTI = "#2a78d6"

# ---- bande usate -------------------------------------------------------------
abl = json.load(open(f'{REP}/rt_fbc_comparison/ablations.json'))
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (ds, name, labels) in zip(axes, SETS):
    c = abl[f'{ds}|ri']['conditions']
    clean = c['clean']['accuracy']
    d = [100 * (c[f'trace_{k}_off']['accuracy'] - clean) for k in range(4)]
    x = np.arange(4)
    ax.bar(x, d, width=0.62, color=BAND, edgecolor=[INK2 if k == 3 else BAND[k] for k in range(4)],
           linewidth=0.6, zorder=2)
    for xi, v in zip(x, d):
        ax.text(xi, v - 0.8, r"$%s$" % num(v, 1), ha='center', va='top', fontsize=7.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"banda {k + 1}\n{lab}" for k, lab in enumerate(labels)], fontsize=7.2)
    ax.set_xlabel("intervallo di ritardi della banda [passi]", fontsize=7.5)
    ax.set_ylim(-27, 1.5)
    ax.yaxis.set_major_formatter(comma(0))
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(name, fontsize=8.5)
axes[0].set_ylabel("variazione di accuratezza [punti]", fontsize=7.5)
save(fig, 'cap5_bande_usate')

# ---- ordine dei passi ----------------------------------------------------------
rows = list(csv.DictReader(open(f'{REP}/c0_ri_diagnostics/csv/E2.csv')))
acc = {(r['dataset'], r['modello'], r['condizione']): float(r['accuracy']) for r in rows}
COND = [('original', 'originale'), ('reverse', 'invertito'), ('shuffle_mean', 'rimescolato')]
MOD = [('C0', 'modello di riferimento', NEUTRAL), ('B-', 'IRM antisimmetrico', ANTI)]
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (ds, name, _) in zip(axes, SETS):
    x = np.arange(len(COND))
    for j, (m, lab, col) in enumerate(MOD):
        v = [acc[(ds, m, c)] for c, _ in COND]
        xs = x + (j - 0.5) * 0.34
        ax.bar(xs, v, width=0.32, color=col, label=lab, zorder=2)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 1.2, r"$%s$" % num(vi, 1), ha='center', va='bottom', fontsize=6.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in COND], fontsize=7.5)
    ax.set_ylim(0, 78)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(name, fontsize=8.5)
axes[0].set_ylabel(r"accuratezza sul test [\%]", fontsize=7.5)
axes[0].yaxis.set_major_formatter(comma(0))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.53, -0.1), ncol=2, fontsize=7.2)
save(fig, 'cap5_ordine')
