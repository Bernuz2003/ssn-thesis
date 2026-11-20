"""Percorso verso i modelli finali (sezione 6.5): figura cap6_percorso.

Accuratezza sul test ufficiale lungo la catena di scelte, una leva per riga, per DVS-Lip e THU-EACT-50-CHL.
Linee tratteggiate: migliore SNN pubblicata (Spiking MSTP su DVS-Lip, Motion SNN su THU-EACT-50-CHL) e, su
THU-EACT-50-CHL, migliore metodo senza dati esterni (HoloEv-Net-Base). A 2 M su THU-EACT-50-CHL, media di tre
addestramenti. Fonte: finali6.py e compressione6.py (logit di artifacts/reports/b_minus_final_inference).
"""
import numpy as np
from common6 import *
from finali6 import test, group
from compressione6 import worker


def comp_mean(ws):
    return float(np.mean([worker(w)['test_comp'] for w in ws]))


THU2 = ['thu_b_2m', 'thu_b_2m_s43', 'thu_b_2m_s44']
STEPS = {
    'dvslip': [("modello di riferimento", group('lip_c0')['acc']),
               ("IRM", group('lip_b')['acc']),
               ("+ relazioni locali", group('lip_prepool')['acc']),
               ("+ distillazione", group('lip_prepool_kd4_05')['acc']),
               ("1 M di parametri", group('lip_prepool_1m_kd4')['acc']),
               ("2 M di parametri", group('lip_prepool_2m_kd4')['acc']),
               ("compresso", comp_mean(['lip_prepool_2m_kd4']))],
    'thu_eact_50_chl': [("modello di riferimento", group('thu_c0')['acc']),
                        ("IRM", group('thu_b')['acc']),
                        ("1 M di parametri", group('thu_b_1m')['acc']),
                        ("2 M di parametri", group('thu_b_2m')['acc']),
                        ("compresso", comp_mean(THU2))],
}
REF = {'dvslip': [(60.20, "Spiking MSTP", 'top')],
       'thu_eact_50_chl': [(47.3, "migliore SNN", 'bottom'), (69.85, "HoloEv-Net-Base", 'top')]}
XL = {'dvslip': (46, 64), 'thu_eact_50_chl': (44, 78)}

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw=dict(wspace=0.75))
for ax, ds in zip(axes, ('dvslip', 'thu_eact_50_chl')):
    labs = [l for l, _ in STEPS[ds]]; v = np.array([a for _, a in STEPS[ds]])
    y = np.arange(len(v))[::-1]
    col = DSCOL[ds]
    ax.plot(v, y, '-', color=tint(col, 0.5), lw=1.2, zorder=2)
    ax.plot(v[:1], y[:1], 'o', ms=5.5, mfc='white', mec=NEUTRAL, mew=1.2, zorder=3)
    ax.plot(v[1:], y[1:], 'o', ms=5.5, color=col, zorder=3)
    span = 0.13 * (XL[ds][1] - XL[ds][0])
    for vi, yi in zip(v, y):
        left = any(vi + 0.05 < r[0] < vi + span for r in REF[ds])  # l'etichetta non attraversa le linee di riferimento
        ax.annotate(r"$%s$" % num(vi, 1), (vi, yi), xytext=(-6 if left else 6, 0), textcoords='offset points',
                    ha='right' if left else 'left', va='center', fontsize=6.6, color=INK)
    for xv, lab, where in REF[ds]:
        ax.axvline(xv, color=INK2, lw=0.8, ls='--', zorder=1)
        if where == 'top':
            ax.text(xv - 0.4, len(v) - 0.5, lab, ha='right', va='center', fontsize=6.2, color=INK2)
        else:
            ax.text(xv + 0.6, -0.35, lab, ha='left', va='center', fontsize=6.2, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlim(*XL[ds]); ax.set_ylim(-0.7, len(v) - 0.2)
    ax.xaxis.set_major_formatter(comma(0))
    ax.set_xlabel(r"accuratezza sul test [\%]", fontsize=7.5)
    ax.set_title(DSNAME[ds], fontsize=8.5)
    ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    print(ds, [f"{l}: {a:.2f}" for l, a in STEPS[ds]])
save(fig, 'cap6_percorso')
