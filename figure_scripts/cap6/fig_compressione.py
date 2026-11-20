"""Compressione dopo l'addestramento (sezione 6.3): figura cap6_compressione, quattro pannelli.

(a) Differenza di Macro-F1 di validazione rispetto al modello denso al variare del rango q della base condivisa, per
    l'IRM con 0,5, 1 e 2 milioni di parametri (primo addestramento a 2 M) e, su DVS-Lip, per l'IRM distillato compatto
    (cinque insegnanti) e da 2 M (primo addestramento). Banda grigia: tolleranza di 0,5 punti; cerchio grande: rango scelto.
(b) MAC per frame del sito del secondo patch embedding: mixer a ritardi del modello di riferimento, IRM denso e IRM
    compresso al rango scelto (primo addestramento a 2 M, che e' anche il rango piu' grande fra i tre).
Fonte: compressione6.py (report di artifacts/reports/b_minus_final_inference).
"""
import numpy as np
from common6 import *
from compressione6 import worker, cost, load

RAMP = ['#9dbfe8', '#2a78d6', '#0d3d78']
RAMP_T = ['#f6b98f', '#eb6834', '#a63d10']
CURVES = {'dvslip': [('IRM, 0,5 M', 'lip_b', RAMP[0], '-'), ('IRM, 1 M', 'lip_b_1m', RAMP[1], '-'),
                     ('IRM, 2 M', 'lip_b_2m', RAMP[2], '-'),
                     ('IRM distillato, 0,5 M', 'lip_kd5', '#7a7a7a', '--'),
                     ('IRM distillato, 2 M', 'lip_kd4_2m', '#3d3d3d', '--')],
          'thu_eact_50_chl': [('IRM, 0,5 M', 'thu_b', RAMP_T[0], '-'), ('IRM, 1 M', 'thu_b_1m', RAMP_T[1], '-'),
                              ('IRM, 2 M', 'thu_b_2m', RAMP_T[2], '-')]}
BUDGET = {'dvslip': [('0,5 M', 'lip_c0', 'lip_b'), ('1 M', 'lip_c0_1m', 'lip_b_1m'), ('2 M', 'lip_c0_2m', 'lip_b_2m')],
          'thu_eact_50_chl': [('0,5 M', 'thu_c0', 'thu_b'), ('1 M', 'thu_c0_1m', 'thu_b_1m'),
                              ('2 M', 'thu_c0_2m', 'thu_b_2m')]}

fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), gridspec_kw=dict(wspace=0.22, hspace=0.55))
for j, ds in enumerate(('dvslip', 'thu_eact_50_chl')):
    ax = axes[0][j]
    ax.axhspan(-0.5, 0, color='#e6e6e6', lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for lab, w, col, ls in CURVES[ds]:
        o = worker(w)
        r = o['ranks']; y = [o['val_curve'][k] for k in r]
        ax.plot(r, y, ls=ls, marker='o', ms=3, lw=1.3, color=col, label=lab, zorder=2)
        if o['q'] is not None:
            ax.plot(o['q'], o['val_curve'][o['q']], 'o', ms=8, mfc=col, mec=INK, mew=0.8, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([8, 16, 32, 64, 128]); ax.set_xticklabels(['8', '16', '32', '64', '128']); ax.minorticks_off()
    ax.set_ylim(-9, 1.5)
    ax.set_xlabel("rango $q$ della base condivisa", fontsize=7.5)
    ax.yaxis.set_major_formatter(comma(0))
    ax.set_title(DSNAME[ds], fontsize=8.5)
    ax.legend(loc='lower right', fontsize=6.4, frameon=False, handlelength=2.2)
axes[0][0].set_ylabel("differenza di Macro-F1\nrispetto al denso [punti]", fontsize=7.5)

for j, ds in enumerate(('dvslip', 'thu_eact_50_chl')):
    ax = axes[1][j]
    x = np.arange(3)
    cols = [NEUTRAL, tint(DSCOL[ds], 0.45), DSCOL[ds]]
    labs = ['mixer a ritardi', 'IRM denso', 'IRM compresso']
    for i, (size, c0, b) in enumerate(BUDGET[ds]):
        m = cost(load(c0)['arms']['dense']['profile'])['module_mac'] / 1e6
        o = worker(b)
        v = [m, o['dense']['module_mac'] / 1e6, o['comp']['module_mac'] / 1e6]
        for k in range(3):
            xi = i + (k - 1) * 0.27
            ax.bar(xi, v[k], width=0.25, color=cols[k], label=labs[k] if i == 0 else None, zorder=2)
            ax.text(xi, v[k] * 1.02 + 0.3, r"$%s$" % num(v[k], 1), ha='center', va='bottom', fontsize=6.0,
                    color=INK)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _, _ in BUDGET[ds]], fontsize=7.5)
    ax.set_xlabel("parametri della rete", fontsize=7.5)
    ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
    ax.yaxis.set_major_formatter(comma(0))
    ax.set_ylim(0, (22 if ds == 'dvslip' else 125))
    ax.legend(loc='upper left', fontsize=6.4, frameon=False)
axes[1][0].set_ylabel("MAC del sito per passo [M]", fontsize=7.5)
for ax, tag in [(axes[0][0], '(a)'), (axes[1][0], '(b)')]:
    ax.text(-0.2, 1.1, tag, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')
save(fig, 'cap6_compressione')
