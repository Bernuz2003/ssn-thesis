"""Capacita' (sezione 6.2): figura cap6_capacita.

Accuratezza sul test ufficiale (valutazione a posteriori) del modello di riferimento e dell'IRM antisimmetrico con
128, 192 e 256 canali, in funzione dei parametri (scala logaritmica). Su DVS-Lip la variante migliore e' l'IRM
con le relazioni locali prima del pooling, addestrato in modo standard e con distillazione da quattro insegnanti. A 2 M: media degli
addestramenti ripetuti, senza barre di deviazione (IRM tre, riferimento THU tre, IRM distillato su DVS-Lip tre; riferimento DVS-Lip uno).
Su DVS-Lip anche il riferimento con distillazione da quattro insegnanti a 2 M (stella vuota).
Fonte: finali6.py (logit di artifacts/reports/b_minus_final_inference).
"""
import numpy as np
from common6 import *
from finali6 import group, PARAMS

GRAY = NEUTRAL
SER = {
    'dvslip': [('modello di riferimento', ['lip_c0', 'lip_c0_1m', 'lip_c0_2m'], GRAY, 'o', '-'),
               ('IRM e relazioni locali', ['lip_prepool', 'lip_prepool_1m', 'lip_prepool_2m'], DSCOL['dvslip'], 'o', '-'),
               ('IRM e relazioni locali, distillato', ['lip_prepool_kd4_05', 'lip_prepool_1m_kd4', 'lip_prepool_2m_kd4'],
                DSCOL['dvslip'], '*', '--')],
    'thu_eact_50_chl': [('modello di riferimento', ['thu_c0', 'thu_c0_1m', 'thu_c0_2m'], GRAY, 'o', '-'),
                        ('IRM antisimmetrico', ['thu_b', 'thu_b_1m', 'thu_b_2m'], DSCOL['thu_eact_50_chl'], 'o', '-')],
}
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw=dict(wspace=0.22))
for ax, ds in zip(axes, ('dvslip', 'thu_eact_50_chl')):
    vals = {}
    for lab, ws, col, mk, ls in SER[ds]:
        g = [group(w) for w in ws]
        x = np.array([PARAMS[w] / 1e6 for w in ws]); y = np.array([v['acc'] for v in g])
        e = np.array([v['acc_sd'] for v in g])
        vals[lab] = y
        ax.plot(x, y, ls=ls, color=col, lw=1.4, zorder=2)
        ax.plot(x, y, mk, ms=7 if mk == '*' else 5, color=col,
                mfc=col if mk == '*' else ('white' if col == GRAY else col), mec=col, mew=1.2, zorder=3, label=lab)
        print(ds, lab, np.round(y, 2), np.round(e, 2))
    if ds == 'dvslip':
        g = group('lip_c0_2m_kd4')
        xr = PARAMS['lip_c0_2m_kd4'] / 1e6 * 0.93  # spostato a sinistra per non coprire l'IRM distillato
        ax.plot(xr, g['acc'], '*', ms=7, mfc='white', mec=GRAY, mew=1.1, zorder=3,
                label='riferimento con distillazione')
        print(ds, 'riferimento con distillazione', round(g['acc'], 2))
    best = SER[ds][1][0]  # variante migliore con la ricetta standard
    d = vals[best] - vals['modello di riferimento']
    xs = [PARAMS[w] / 1e6 for w in SER[ds][1][1]]
    for xi, yi, yr, di in zip(xs, vals[best], vals['modello di riferimento'], d):
        ax.plot([xi * 1.035] * 2, [yr, yi], color=INK2, lw=0.6, zorder=1)
        lab = (r"$+%s$" % num(di, 1)) if di >= 0 else (r"$%s$" % num(di, 1))
        ax.annotate(lab, (xi * 1.05, (yi + yr) / 2), ha='left', va='center', fontsize=6.8, color=INK2)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 2]); ax.set_xticklabels(['0,5', '1', '2'])
    ax.minorticks_off()
    ax.set_xlim(0.42, 2.6)
    ax.set_xlabel("parametri [milioni]", fontsize=7.5)
    ax.set_title(DSNAME[ds], fontsize=8.5)
    ax.yaxis.set_major_formatter(comma(0))
    ax.legend(loc='upper left' if ds == 'dvslip' else 'lower right', fontsize=6.6, frameon=False)
axes[0].set_ylim(47, 67.5); axes[0].set_yticks([48, 52, 56, 60, 64]); axes[1].set_ylim(53, 79)
axes[0].set_ylabel(r"accuratezza sul test [\%]", fontsize=7.5)
save(fig, 'cap6_capacita')
