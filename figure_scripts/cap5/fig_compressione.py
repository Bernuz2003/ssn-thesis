"""Compressione dell'IRM antisimmetrico dopo l'addestramento (sezione 5.7).

(a) energia dei pesi W_k conservata dalla base condivisa di rango q; (b) differenza di Macro-F1 rispetto al
modello denso valutato nella stessa esecuzione dello sweep, sulla validazione di sviluppo. Sola inferenza,
nessun riaddestramento. Fonte: artifacts/reports/b_minus_compression/rank_sweep.json.
Nota: su DVS-Lip lo sweep usa per ora il checkpoint IRM 20261026_112001 (si mostrano solo differenze rispetto
al proprio denso); da aggiornare con lo sweep sul checkpoint di riferimento 20261112_082513.
"""
import json
from matplotlib.lines import Line2D
from common5 import *

SW = json.load(open('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/'
                    'b_minus_compression/rank_sweep.json'))
SETS = [('dvslip', 'DVS-Lip', 64, "#2a78d6", 'o'), ('thu_eact_50_chl', 'THU-EACT-50-CHL', 32, "#eb6834", 's')]

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), gridspec_kw=dict(wspace=0.3))
for ds, name, chosen, col, mk in SETS:
    keys = sorted((k for k in SW if k.startswith(ds + '/q')), key=lambda k: SW[k]['rank'])
    q = [SW[k]['rank'] for k in keys]
    en = [100 * SW[k]['retained_weight_energy'] for k in keys]
    dense = 100 * SW[f'{ds}/dense']['f1']
    df = [100 * SW[k]['f1'] - dense for k in keys]
    for ax, y in zip(axes, (en, df)):
        ax.plot(q, y, color=col, marker=mk, ms=4.2, lw=1.5, label=name, zorder=3)
        i = q.index(chosen)
        ax.plot(q[i], y[i], marker=mk, ms=8.5, color=col, markeredgecolor='white', markeredgewidth=1.2, zorder=4)
    i = q.index(chosen)
    axes[1].annotate(r"$q=%d$" % chosen, (q[i], df[i]), xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=7, color=INK)
axes[0].axhline(100, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
axes[0].set_ylim(45, 103)
axes[0].set_ylabel(r"energia dei pesi conservata [\%]", fontsize=7.5)
axes[0].set_title(r"\textbf{(a)} ricostruzione dei pesi", loc='left', fontsize=8.2)
axes[1].axhline(0, color=INK2, lw=0.8, zorder=1)
axes[1].set_ylim(-3.4, 1.8)
axes[1].yaxis.set_major_formatter(comma(0))
axes[1].set_ylabel("differenza di Macro-F1 [punti]", fontsize=7.5)
axes[1].set_title(r"\textbf{(b)} prestazioni rispetto al modello denso", loc='left', fontsize=8.2)
for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_xticks([8, 16, 32, 48, 64, 96])
    ax.set_xlabel(r"rango $q$ della base condivisa", fontsize=7.5)
handles = [Line2D([], [], color=c, marker=m, ms=4.5, lw=1.5, label=n) for _, n, _, c, m in SETS]
handles.append(Line2D([], [], color=INK2, marker='o', ms=8, ls='', markeredgecolor='white', markeredgewidth=1.2, label='rango scelto'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.52, -0.2), ncol=3, fontsize=7.2)
save(fig, 'cap5_compressione')
