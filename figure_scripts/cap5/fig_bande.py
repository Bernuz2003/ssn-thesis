"""Ritardi del mixer a ritardi e bande della memoria a intervalli (figura 5.2), in tempo fisico.

Un pannello per dataset, con asse lineare che parte da 0 (il presente). Ogni passo e' disegnato come un intervallo
centrato sul proprio ritardo, sicche' il presente occupa la prima mezza cella e la banda 1 comincia subito dopo.
"""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from common5 import *

rows = list(DS)
XMAX = {'dvslip': 450, 'thu_eact_50_chl': 1700}
fig, axes = plt.subplots(len(rows), 1, figsize=(WIDTH, 2.3), gridspec_kw=dict(hspace=0.95))
for ax, (d, C, name) in zip(axes, rows):
    dl, dt = DELAYS[d], DELTA_MS[d]
    ax.add_patch(Rectangle((0, -0.28), 0.5 * dt, 0.56, facecolor=INK, edgecolor='white', linewidth=1.0))
    lo = 0
    for k, up in enumerate(dl):
        x0, x1 = (lo + 0.5) * dt, (up + 0.5) * dt
        ax.add_patch(Rectangle((x0, -0.28), x1 - x0, 0.56, facecolor=BAND[k], edgecolor='white', linewidth=1.0))
        lo = up
    lags = np.arange(1, dl[-1] + 1) * dt
    ax.plot(lags, np.full_like(lags, 0.42), linestyle='', marker='|', markersize=4, color=MUTED)
    ax.plot(np.array(dl) * dt, np.full(len(dl), 0.42), linestyle='', marker='o', markersize=4.5,
            color=INK, markeredgecolor='white', markeredgewidth=0.6)
    ax.text(1.01, 0, r"%d ritardi su %d" % (len(dl), dl[-1]), transform=ax.get_yaxis_transform(),
            va='center', fontsize=7.2, color=INK2)
    ax.set_xlim(0, XMAX[d]); ax.set_ylim(-0.45, 0.62)
    ax.set_yticks([0]); ax.set_yticklabels([name], fontsize=7.5)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='x', labelsize=7)
axes[-1].set_xlabel("distanza nel passato [ms]", fontsize=7.5)
handles = [Rectangle((0, 0), 1, 1, facecolor=INK, edgecolor='white', label='presente'),
           Line2D([], [], linestyle='', marker='o', color=INK, markersize=4.5, label='ritardi del mixer'),
           Line2D([], [], linestyle='', marker='|', color=MUTED, markersize=6, label='passi del passato')]
handles += [Rectangle((0, 0), 1, 1, facecolor=BAND[k], edgecolor='white', label=f'banda {k + 1}') for k in range(4)]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=7, fontsize=6.6,
           handlelength=1.0, columnspacing=0.7, handletextpad=0.35)
save(fig, 'cap5_bande')
