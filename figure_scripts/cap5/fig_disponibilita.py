import numpy as np
from matplotlib.patches import Rectangle
from common5 import *

fig = plt.figure(figsize=(WIDTH, 3.2))
gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.25], hspace=0.62)

# (a) Schema: bande di DVS-Lip a due istanti della sequenza
ax = fig.add_subplot(gs[0])
dl = DELAYS['dvslip']; dK = dl[-1]
band_of = {}
lo = 0
for k, up in enumerate(dl):
    for l in range(lo + 1, up + 1): band_of[l] = k
    lo = up
for row, t in enumerate((5, 30)):
    y = 1.5 - 1.5 * row
    for l in range(1, dK + 1):
        x = -l
        avail = l <= t
        k = band_of[l]
        ax.add_patch(Rectangle((x - 0.5, y - 0.36), 1, 0.72, facecolor=BAND[k] if avail else 'white',
                               edgecolor='white' if avail else MUTED, linewidth=0.8 if avail else 0.6,
                               hatch=None if avail else '////', alpha=1.0))
    ax.add_patch(Rectangle((-0.5 + 1, y - 0.36), 1, 0.72, facecolor=INK, edgecolor='white', linewidth=0.8))
    ax.text(1.0, y, r"$z_t$", ha='center', va='center', fontsize=7.5, color='white')
    lo = 0; ann = []
    for k, up in enumerate(dl):
        n = max(0, min(t, up) - lo); w = up - lo
        ann.append(r"$%d/%d$" % (n, w))
        ax.text(-(lo + up + 1) / 2, y + 0.5 if row == 0 else y - 0.5, ann[-1], ha='center',
                va='bottom' if row == 0 else 'top', fontsize=7, color=INK2)
        lo = up
    ax.text(-dK - 1.2, y, r"$t=%d$" % t, ha='right', va='center', fontsize=8, color=INK)
ax.set_xlim(-dK - 3.4, 1.8); ax.set_ylim(-0.95, 2.35)
ax.set_xticks([-15, -8, -5, -3, -1, 1]); ax.set_xticklabels([r'$15$', r'$8$', r'$5$', r'$3$', r'$1$', r'$0$'])
ax.set_yticks([]); ax.grid(False)
for s in ('left', 'top', 'right'): ax.spines[s].set_visible(False)
ax.set_xlabel("ritardo in passi (DVS-Lip, passi da 27,4\\,ms)", fontsize=8)
ax.set_title(r"\textbf{(a)} passi disponibili in ciascuna banda all'inizio e a regime", loc='left')

# (b) Griglia dataset x banda: quota di passi con storia completa (e vuota)
from matplotlib.colors import LinearSegmentedColormap
ax2 = fig.add_subplot(gs[1])
cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cfe0f6', '#2a78d6', '#0d3d78'])
M = np.zeros((4, 4)); E = np.zeros((4, 4)); names = []
for i, (d, C, name) in enumerate(DS):
    V = load(d, 'VC'); T = V['steps'].astype(int); tot = T.sum(); lo = 0
    for k, up in enumerate(DELAYS[d]):
        E[i, k] = sum(min(Ti, lo + 1) for Ti in T) / tot
        M[i, k] = sum(max(0, Ti - up) for Ti in T) / tot
        lo = up
    names.append(name)
ax2.imshow(M * 100, cmap=cmap, vmin=0, vmax=100, aspect='auto')
for i in range(4):
    for k in range(4):
        dark = M[i, k] > 0.6
        ax2.text(k, i - 0.12, r"$%s\%%$" % num(M[i, k] * 100, 0), ha='center', va='center', fontsize=8.5,
                 color='white' if dark else INK)
        ax2.text(k, i + 0.27, r"vuota $%s\%%$" % num(E[i, k] * 100, 0), ha='center', va='center', fontsize=6.5,
                 color='#e8eef8' if dark else INK2)
ax2.set_xticks(range(4)); ax2.set_xticklabels(["banda 1 (vicina)", "banda 2", "banda 3", "banda 4 (lontana)"], fontsize=7.5)
ax2.set_yticks(range(4)); ax2.set_yticklabels(names, fontsize=7.5)
ax2.tick_params(length=0); ax2.grid(False)
for sp in ax2.spines.values(): sp.set_visible(False)
ax2.set_title(r"\textbf{(b)} passi con storia completa nella banda, validazione di sviluppo", loc='left')
save(fig, 'cap5_disponibilita')
