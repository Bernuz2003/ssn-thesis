import numpy as np
from matplotlib.patches import Rectangle
from style import *

z = np.load('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/data/DvsGesture/events/train/6/user17_led__06.npz')
t = z['t'].astype(np.int64) / 1e6  # secondi dall'inizio del segmento
N = len(t); t0, t1 = t[0], t[-1]
WIN, DT_E0, DT_VC, T = 20.0, 0.5, 0.164358, 40

# Partizioni
e0_edges = np.arange(T + 1) * DT_E0
e0_counts = np.histogram(t, e0_edges)[0]
q, r = divmod(N, T)
sizes = np.array([q + 1] * r + [q] * (T - r)); starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
fec_lo = t[starts]; fec_hi = np.append(t[starts[1:]], t1)
Tvc = int(np.ceil((round((t1 - t0) * 1e6) + 1) / (DT_VC * 1e6)))
vc_edges = t0 + np.arange(Tvc + 1) * DT_VC
vc_counts = np.histogram(t, vc_edges)[0]
print('E0 occupied', (e0_counts > 0).sum(), 'VC steps', Tvc, 'VC empty', (vc_counts == 0).sum(), 'N', N, 'dur', t1 - t0)

fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 3.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.1, 0.62, 0.62, 0.62], hspace=0.28))
ax = axes[0]
rate_edges = np.arange(0, WIN + 0.01, 0.01)
rate = np.histogram(t, rate_edges)[0] / 0.01 / 1e3
ax.fill_between(rate_edges[:-1], rate, step='post', color=NEUTRAL, alpha=0.55, linewidth=0)
ax.axvspan(t1, WIN, color=CLOCK['E0']['color'], alpha=0.07, linewidth=0)
ax.axvline(t1, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
ax.text(t1 + 0.25, rate.max() * 0.92, r"ultimo evento, $%s$\,s" % num(t1, 1), color=INK2, fontsize=8, va='top')
ax.text((t1 + WIN) / 2, rate.max() * 0.55, "coda della finestra fissa\n(nessun evento)", color=CLOCK['E0']['color'], fontsize=8, ha='center', va='center')
ax.set_ylabel(r"eventi/ms")
ax.set_ylim(0, rate.max() * 1.05)
ax.yaxis.set_major_formatter(comma(0))
ax.set_title(r"Un gesto di DVS128 Gesture (braccio sinistro, rotazione antioraria): lo stesso flusso, tre clock", loc='left', fontsize=9, x=0.0)

def cells(ax, lo, hi, counts, key, label):
    c = CLOCK[key]['color']
    for a, b, n in zip(lo, hi, counts):
        ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=c, alpha=0.85 if n > 0 else 0.13,
                               edgecolor='white', linewidth=0.8))
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
    for s in ('left', 'bottom'): ax.spines[s].set_visible(False)
    ax.text(8.75, 0.5, label, ha='left', va='center', fontsize=8, color=INK)

cells(axes[1], e0_edges[:-1], e0_edges[1:], e0_counts, 'E0', r"\textbf{Finestra fissa}, 40 bin da 500\,ms, 16 con eventi")
cells(axes[2], fec_lo, fec_hi, sizes, 'FEC', r"\textbf{Eventi costanti}, 40 gruppi da circa %s eventi" % f"{q:,}".replace(',', '\\,'))
cells(axes[3], vc_edges[:-1], vc_edges[1:], vc_counts, 'VC', r"\textbf{Lunghezza variabile}, %d bin da 164\,ms" % Tvc)
for a in axes[1:]:
    a.tick_params(left=False)
axes[-1].spines['bottom'].set_visible(True)
axes[-1].set_xlabel("tempo dall'inizio del segmento [s]")
axes[-1].set_xlim(0, WIN)
from matplotlib.ticker import MultipleLocator
axes[-1].xaxis.set_major_locator(MultipleLocator(2))
axes[-1].xaxis.set_major_formatter(comma(0))
save(fig, 'cap4_tre_clock')
