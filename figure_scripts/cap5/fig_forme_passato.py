"""Tre modi di riassumere il passato (sezione 5.5): peso di ciascun passo passato nei quattro riassunti.

(a) bande di DVS-Lip [1,2], [3,4], [5,7], [8,15], media uniforme;
(b) tracce esponenziali con gli orizzonti appresi dalle tracce statiche su DVS-Lip (0,13 / 0,50 / 1,36 / 4,98 s,
    ricavati dai checkpoint), in passi da 27,4 ms; ogni curva e' normalizzata al proprio massimo;
(c) memoria di consolidamento: quattro segmenti che ripartiscono tutto il passato, con bordi che dipendono dal
    contenuto; due configurazioni illustrative, non misurate.
"""
import numpy as np
from matplotlib.patches import Rectangle
from common5 import *

L = 32
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.0), sharey=False, gridspec_kw=dict(wspace=0.28))

# (a) bande
ax = axes[0]
lo = 0
for k, up in enumerate([2, 4, 7, 15]):
    ax.add_patch(Rectangle((lo + 0.5, 0), up - lo, 1.0, fc=BAND[k], ec='white', lw=0.8))
    lo = up
ax.set_ylim(0, 1.3)
ax.text(L * 0.97, 1.15, "una media per banda,\nentro ritardi fissi", ha='right', va='top', fontsize=6.6, color=INK2)
ax.set_title(r"\textbf{(a)} bande a ritardi fissi", loc='left', fontsize=8.2)

# (b) tracce
ax = axes[1]
lags = np.arange(1, L + 1)
for k, hor in enumerate([0.13, 0.50, 1.36, 4.98]):
    tau = hor / 0.027422
    w = np.exp(-(lags - 1) / tau)
    ax.plot(lags, w, color=BAND[k] if k < 3 else '#9dbfe8', lw=1.6, drawstyle='steps-mid')
ax.set_ylim(0, 1.3)
ax.text(L * 0.97, 1.15, "tutte partono dal\npasso precedente", ha='right', va='top', fontsize=6.6, color=INK2)
ax.set_title(r"\textbf{(b)} tracce esponenziali", loc='left', fontsize=8.2)

# (c) consolidamento: due configurazioni illustrative
ax = axes[2]
for row, edges in enumerate(([3, 9, 22, L], [1, 12, 17, L])):
    y0 = 0.68 - 0.58 * row
    lo = 0
    for k, up in enumerate(edges):
        ax.add_patch(Rectangle((lo + 0.5, y0), up - lo, 0.42, fc=BAND[k], ec='white', lw=0.8))
        lo = up
ax.text(L * 0.5, 1.2, "bordi decisi dal contenuto", ha='center', va='center', fontsize=6.6, color=INK2)
ax.set_ylim(0, 1.3)
ax.set_title(r"\textbf{(c)} consolidamento", loc='left', fontsize=8.2)

for i, ax in enumerate(axes):
    ax.set_xlim(0.5, L + 0.5)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_xlabel("ritardo [passi]", fontsize=7.5)
    ax.grid(axis='x', visible=False)
    if i == 0:
        ax.set_yticks([0, 1]); ax.set_ylabel("peso", fontsize=7.5)
    else:
        ax.set_yticks([])
    if i == 2:
        ax.spines['left'].set_visible(False)
save(fig, 'cap5_forme_passato')
