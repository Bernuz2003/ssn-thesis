"""Compressione dopo l'addestramento (sezione 6.3): figure cap6_compressione e cap6_compressione_costi.

cap6_compressione
    Differenza di Macro-F1 di validazione rispetto al modello denso al variare del rango q della base condivisa, per
    il modello finale di ciascun dataset con 0,5, 1 e 2 milioni di parametri: su THU-EACT-50-CHL l'IRM (primo
    addestramento a 2 M), su DVS-Lip l'IRM con relazioni locali e distillazione (si comprime solo l'IRM). Banda grigia: tolleranza di 0,5 punti; cerchio grande:
    rango scelto.
cap6_compressione_costi
    MAC per passo (riga 1) e stato persistente in valori (riga 2) del sito del secondo patch embedding: mixer a ritardi
    del modello di riferimento, IRM denso e IRM compresso al rango scelto (a 2 M il primo addestramento, che ha il
    rango piu' grande fra i tre su THU-EACT-50-CHL e uno dei ranghi piu' grandi su DVS-Lip).
Fonte: compressione6.py (report di artifacts/reports/b_minus_final_inference).
"""
import numpy as np
from common6 import *
from compressione6 import worker, cost, load

RAMP = ['#9dbfe8', '#2a78d6', '#0d3d78']
RAMP_T = ['#f6b98f', '#eb6834', '#a63d10']
CURVES = {'dvslip': [('modello finale, 0,5 M', 'lip_prepool_kd4_05', RAMP[0], '-'),
                     ('modello finale, 1 M', 'lip_prepool_1m_kd4', RAMP[1], '-'),
                     ('modello finale, 2 M', 'lip_prepool_2m_kd4', RAMP[2], '-')],
          'thu_eact_50_chl': [('IRM, 0,5 M', 'thu_b', RAMP_T[0], '-'), ('IRM, 1 M', 'thu_b_1m', RAMP_T[1], '-'),
                              ('IRM, 2 M', 'thu_b_2m', RAMP_T[2], '-')]}
BUDGET = {'dvslip': [('0,5 M', 'lip_c0', 'lip_prepool_kd4_05'), ('1 M', 'lip_c0_1m', 'lip_prepool_1m_kd4'),
                     ('2 M', 'lip_c0_2m', 'lip_prepool_2m_kd4')],
          'thu_eact_50_chl': [('0,5 M', 'thu_c0', 'thu_b'), ('1 M', 'thu_c0_1m', 'thu_b_1m'),
                              ('2 M', 'thu_c0_2m', 'thu_b_2m')]}

# ---- ranghi ----------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), gridspec_kw=dict(wspace=0.22))
for ax, ds in zip(axes, ('dvslip', 'thu_eact_50_chl')):
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
axes[0].set_ylabel("differenza di Macro-F1\nrispetto al denso [punti]", fontsize=7.5)
save(fig, 'cap6_compressione')

# ---- costi del sito ------------------------------------------------------------------------------------------
ROWS = [('module_mac', 1e6, 1, "MAC del sito\nper passo [milioni]", {'dvslip': 22, 'thu_eact_50_chl': 125}),
        ('module_state', 1e3, 0, "stato del sito\n[migliaia di valori]", {'dvslip': 330, 'thu_eact_50_chl': 3000})]
fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6), gridspec_kw=dict(wspace=0.24, hspace=0.42))
for r, (key, scale, dec, ylab, ylim) in enumerate(ROWS):
    for j, ds in enumerate(('dvslip', 'thu_eact_50_chl')):
        ax = axes[r][j]
        cols = [NEUTRAL, tint(DSCOL[ds], 0.45), DSCOL[ds]]
        labs = ['mixer a ritardi', 'IRM denso', 'IRM compresso']
        for i, (size, c0, b) in enumerate(BUDGET[ds]):
            o = worker(b)
            v = [cost(load(c0)['arms']['dense']['profile'])[key] / scale, o['dense'][key] / scale,
                 o['comp'][key] / scale]
            for k in range(3):
                xi = i + (k - 1) * 0.27
                ax.bar(xi, v[k], width=0.25, color=cols[k], label=labs[k] if i == 0 else None, zorder=2)
                ax.text(xi, v[k] + 0.012 * ylim[ds], r"$%s$" % num(v[k], dec), ha='center', va='bottom',
                        fontsize=5.8, color=INK)
            print(ds, key, size, [round(x, 2) for x in v],
                  'compresso/mixer %.0f%%' % (100 * v[2] / v[0]), 'compresso/denso %.0f%%' % (100 * v[2] / v[1]))
        ax.set_xticks(np.arange(3)); ax.set_xticklabels([s for s, _, _ in BUDGET[ds]], fontsize=7.5)
        ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
        ax.yaxis.set_major_formatter(comma(0))
        ax.set_ylim(0, ylim[ds])
        if r == 0:
            ax.set_title(DSNAME[ds], fontsize=8.5)
            ax.legend(loc='upper left', fontsize=6.4, frameon=False)
        else:
            ax.set_xlabel("parametri della rete", fontsize=7.5)
        if j == 0:
            ax.set_ylabel(ylab, fontsize=7.5)
for ax, tag in [(axes[0][0], '(a)'), (axes[1][0], '(b)')]:
    ax.text(-0.24, 1.08, tag, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')
save(fig, 'cap6_compressione_costi')
