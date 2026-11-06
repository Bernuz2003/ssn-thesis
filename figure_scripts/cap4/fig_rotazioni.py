import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style import *
from load import load
rec = np.load('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/dvsgesture/mechanistic/dvsgesture_c0_fixed_event_count__20261011_134033_534484__seed42/record.npz', allow_pickle=True)
E = load('dvsgesture', 'E0'); V = load('dvsgesture', 'VC')
data = {'E0': (E['targets'], E['pred']), 'FEC': (rec['targets'], rec['logits'].argmax(-1)), 'VC': (V['targets'], V['pred'])}
ROT = [3, 4, 5, 6]
labels = [r"destro $\circlearrowright$", r"destro $\circlearrowleft$", r"sinistro $\circlearrowright$", r"sinistro $\circlearrowleft$"]
cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cfe0f6', '#2a78d6', '#0d3d78'])
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), gridspec_kw=dict(wspace=0.16))
for ax, key in zip(axes, ('E0', 'FEC', 'VC')):
    t, p = data[key]
    M = np.zeros((4, 5), int)
    for i, c in enumerate(ROT):
        m = t == c
        for j, d in enumerate(ROT):
            M[i, j] = np.sum(p[m] == d)
        M[i, 4] = np.sum(~np.isin(p[m], ROT))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=22, aspect='auto')
    for i in range(4):
        for j in range(5):
            v = M[i, j]
            if v == 0: continue
            swap = (i, j) in ((0, 1), (1, 0), (2, 3), (3, 2))
            ax.text(j, i, r"\textbf{%d}" % v if swap else str(v), ha='center', va='center', fontsize=8,
                    color='white' if v > 12 else INK)
    for (i, j) in ((0, 1), (1, 0), (2, 3), (3, 2)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=CLOCK['FEC']['color'] if False else INK2, linewidth=0.8, linestyle=(0, (2, 1.5))))
    sw = sum(M[i, j] for i, j in ((0, 1), (1, 0), (2, 3), (3, 2)))
    ax.set_title(CLOCK[key]['label'], fontsize=8.5)
    ax.set_xlabel(r"\textbf{%d} inversioni di verso" % sw, fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels([r"D$\circlearrowright$", r"D$\circlearrowleft$", r"S$\circlearrowright$", r"S$\circlearrowleft$", "altro"], fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels if key == 'E0' else [], fontsize=7.5)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    print(key, M.tolist(), 'swaps', sw)
axes[0].set_ylabel("classe vera")
save(fig, 'cap4_rotazioni')
