"""Stato dell'arte su THU-EACT-50-CHL (sezione 6.5): figura cap6_stato_arte.

Accuratezza top-1 sul test ufficiale in funzione dei parametri (scala logaritmica). Letteratura come riportata nei
lavori citati (verificata sulle versioni arXiv il 19/11/2026): SMV-EAR tab. 2 (EV-ACT, EventMG, EventMamba, SMV-EAR),
HoloEv-Net tab. I (MVFNet, HoloEv-Net-Small e -Base), Norrby et al. tab. 1 (EventBind+POKER, ViT-Base con
pre-training auto-supervisionato sul solo THU, ViT-Tiny pre-addestrato su DailyDVS-200). Le due SNN pubblicate (HMAX
SNN 32,7 e Motion SNN 47,3) non hanno un numero di parametri dichiarato: la migliore e' indicata da una linea.
Modelli di questa tesi: finali6.py (IRM antisimmetrico 0,5/1/2 M; a 2 M media di tre addestramenti).
"""
import numpy as np
from common6 import *
from finali6 import group, PARAMS
from compressione6 import worker

LIT = [  # (nome, parametri M, accuratezza, pre-training esterno)
    ('EV-ACT', 21.3, 58.5, False), ('EventMG', 0.77, 58.8, False), ('EventMamba', 0.91, 59.4, False),
    ('MVFNet', 23.6, 59.56, False), ('SMV-EAR', 23.5, 66.7, False), ('HoloEv-Net-Small', 4.4, 67.10, False),
    ('HoloEv-Net-Base', 12.5, 69.85, False), ('ViT-Base, MAE sul solo THU', 85.8, 68.6, False),
    ('EventBind+POKER', 635, 63.7, True), ('ViT-Tiny, DailyDVS-200', 5.5, 78.7, True),
]
OFF = {'EV-ACT': (-5, -9, 'right'), 'EventMG': (2, -12, 'left'), 'EventMamba': (5, 2, 'left'),
       'MVFNet': (5, 2, 'left'), 'SMV-EAR': (5, -3, 'left'), 'HoloEv-Net-Small': (5, -9, 'left'),
       'HoloEv-Net-Base': (0, 6, 'center'), 'ViT-Base, MAE sul solo THU': (6, -2, 'left'),
       'EventBind+POKER': (0, -11, 'center'), 'ViT-Tiny, DailyDVS-200': (7, -2, 'left')}
ORANGE = DSCOL['thu_eact_50_chl']

fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
ax.axhline(47.3, color=INK2, lw=0.8, ls=':', zorder=1)
ax.text(0.42, 47.8, r"migliore SNN pubblicata, $47{,}3$\,\%", ha='left', va='bottom', fontsize=6.6, color=INK2)
for name, p, a, ext in LIT:
    ax.plot(p, a, '^' if ext else 'o', ms=5.5, mfc='white' if ext else '#9a9a9a', mec='#6f6f6f', mew=1.0, zorder=3)
    dx, dy, ha = OFF[name]
    ax.annotate(name, (p, a), xytext=(dx, dy), textcoords='offset points', fontsize=6.3, color=INK2, ha=ha)
irm = [group(w) for w in ('thu_b', 'thu_b_1m', 'thu_b_2m')]
xi = [PARAMS[w] / 1e6 for w in ('thu_b', 'thu_b_1m', 'thu_b_2m')]
ax.plot(xi, [g['acc'] for g in irm], '-o', color=ORANGE, ms=5, lw=1.5, zorder=5,
        label='IRM antisimmetrico (questa tesi)')
for x, g in zip(xi, irm):
    print(f"{x:.2f} M  {g['acc']:.2f}")
ax.plot([], [], 'o', ms=5, mfc='#9a9a9a', mec='#6f6f6f', label='altri metodi, senza dati esterni')
ax.plot([], [], '^', ms=5, mfc='white', mec='#6f6f6f', label='altri metodi, con pre-training esterno')
ax.set_xscale('log')
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
ax.set_xticklabels(['0,5', '1', '2', '5', '10', '20', '50', '100', '200', '500'])
ax.minorticks_off()
ax.set_xlim(0.38, 1200); ax.set_ylim(45, 81)
ax.yaxis.set_major_formatter(comma(0))
ax.set_xlabel("parametri [milioni, scala logaritmica]", fontsize=7.5)
ax.set_ylabel(r"accuratezza sul test [\%]", fontsize=7.5)
h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in ('IRM antisimmetrico (questa tesi)', 'altri metodi, senza dati esterni',
                              'altri metodi, con pre-training esterno')]
ax.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3,
          fontsize=6.3, frameon=False, columnspacing=0.9, handletextpad=0.4)
save(fig, 'cap6_stato_arte')
