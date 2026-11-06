import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from common5 import *

fig, ax = plt.subplots(figsize=(WIDTH, 2.35))
rows = list(DS)[::-1]
for yi, (d, C, name) in enumerate(rows):
    dl, dt = DELAYS[d], DELTA_MS[d]
    lo = 0
    for k, up in enumerate(dl):
        x0, x1 = (lo + 0.5) * dt, (up + 0.5) * dt
        ax.add_patch(Rectangle((x0, yi - 0.28), x1 - x0, 0.56, facecolor=BAND[k], edgecolor='white', linewidth=1.0))
        lo = up
    lags = np.arange(1, dl[-1] + 1) * dt
    ax.plot(lags, np.full_like(lags, yi + 0.36), linestyle='', marker='|', markersize=4, color=MUTED)
    ax.plot(np.array(dl) * dt, np.full(len(dl), yi + 0.36), linestyle='', marker='o', markersize=4.5,
            color=INK, markeredgecolor='white', markeredgewidth=0.6)
    ax.text(1.01, yi, r"%d ritardi su %d" % (len(dl), dl[-1]), transform=ax.get_yaxis_transform(),
            va='center', fontsize=7.5, color=INK2)
ax.set_xscale('log')
ax.set_xlim(18, 5200)
ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000])
ax.set_xticklabels(['20', '50', '100', '200', '500', '1000', '2000', '5000'])
ax.minorticks_off()
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, _, n in rows])
ax.set_ylim(-0.6, len(rows) - 0.25)
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', visible=False)
ax.set_xlabel("distanza nel passato [ms], scala logaritmica")
handles = [Line2D([], [], linestyle='', marker='o', color=INK, markersize=4.5, label='ritardi letti dal mixer a ritardi'),
           Line2D([], [], linestyle='', marker='|', color=MUTED, markersize=6, label='passi del passato')]
handles += [Rectangle((0, 0), 1, 1, facecolor=BAND[k], edgecolor='white', label=f'banda {k + 1}') for k in range(4)]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=6, fontsize=7.2,
           handlelength=1.1, columnspacing=0.9, handletextpad=0.4)
save(fig, 'cap5_bande')
