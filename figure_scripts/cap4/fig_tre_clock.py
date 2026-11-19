"""Una registrazione sintetica suddivisa secondo i tre clock (figura 4.4).

La registrazione è generata qui, come processo di Poisson non omogeneo con passo di 1 ms e seme fisso, e dura 15 s:
due fasi intense con modulazione periodica, tratti quieti, una pausa senza eventi e una coda che si spegne.
I parametri dei clock sono quelli di DVS128 Gesture: finestra fissa di 20 s in 40 bin da 500 ms, 40 gruppi a eventi
costanti sull'intera registrazione, bin a lunghezza variabile da 164,358 ms (tabella 4.2).
"""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from style import *

rng = np.random.default_rng(7)
MS = np.arange(0, 15000) / 1000.0 + 0.0005          # centro di ogni millisecondo [s]


def rate(t):
    """Frequenza degli eventi in eventi/ms."""
    mod = lambda p: 0.75 + 0.25 * np.sin(2 * np.pi * t / p)
    r = np.zeros_like(t)
    r += np.where(t < 1.0, 2 + 8 * t, 0)
    r += np.where((t >= 1.0) & (t < 3.2), 55 * mod(0.6), 0)
    r += np.where((t >= 3.2) & (t < 4.2), 4 + 30 * np.exp(-(t - 3.2) / 0.25), 0)
    r += np.where((t >= 4.2) & (t < 5.6), 3, 0)
    r += np.where((t >= 6.8) & (t < 7.2), 18 * (t - 6.8) / 0.4, 0)
    r += np.where((t >= 7.2) & (t < 10.0), 18 * mod(0.9), 0)
    r += np.where((t >= 10.0) & (t < 12.4), 75 * mod(0.5), 0)
    r += np.where(t >= 12.4, 2 + 9 * np.exp(-(t - 12.4) / 0.9), 0)
    return r


counts = rng.poisson(rate(MS))
t = np.sort(np.concatenate([(i + rng.random(n)) / 1000.0 for i, n in enumerate(counts) if n]))
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
fec_dur = fec_hi - fec_lo
print(f"N {N}  durata {t1 - t0:.2f} s  E0 con eventi {(e0_counts > 0).sum()}  VC passi {Tvc}, vuoti {(vc_counts == 0).sum()}")
print(f"FEC: {q} eventi per gruppo, durate da {1000 * fec_dur.min():.0f} ms a {fec_dur.max():.2f} s")

fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 3.9), sharex=True,
                         gridspec_kw=dict(height_ratios=[2.0, 0.55, 0.55, 0.55], hspace=0.62))
ax = axes[0]
rate_edges = np.arange(0, WIN + 0.01, 0.01)
rate_obs = np.histogram(t, rate_edges)[0] / 0.01 / 1e3
ax.fill_between(rate_edges[:-1], rate_obs, step='post', color=NEUTRAL, alpha=0.6, linewidth=0)
ax.axvspan(t1, WIN, color=CLOCK['E0']['color'], alpha=0.07, linewidth=0)
ax.axvline(t1, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
top = rate_obs.max()
ax.text(t1 + 0.2, top * 0.95, r"ultimo evento, $%s$\,s" % num(t1, 1), color=INK2, fontsize=7.5, va='top')
ax.text((t1 + WIN) / 2, top * 0.5, "coda della finestra fissa\n(nessun evento)", color=CLOCK['E0']['color'],
        fontsize=7.5, ha='center', va='center')
ax.annotate("pausa", xy=(6.2, 1), xytext=(6.2, top * 0.38), ha='center', fontsize=7.5, color=INK2,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.set_ylabel(r"eventi/ms")
ax.set_ylim(0, top * 1.05)
ax.yaxis.set_major_formatter(comma(0))
ax.set_title(r"Una registrazione sintetica di 15\,s e lo stesso flusso suddiviso con tre clock", loc='left',
             fontsize=8.5)


def cells(ax, lo, hi, counts, key, label, lw=0.8):
    c = CLOCK[key]['color']
    for a, b, n in zip(lo, hi, counts):
        ax.add_patch(Rectangle((a, 0), b - a, 1, facecolor=c, alpha=0.85 if n > 0 else 0.13,
                               edgecolor='white', linewidth=lw))
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.set_title(label, loc='left', fontsize=7.5, pad=2.5, color=INK)


cells(axes[1], e0_edges[:-1], e0_edges[1:], e0_counts, 'E0',
      r"\textbf{Finestra fissa}, 40 bin da 500\,ms, %d con eventi" % (e0_counts > 0).sum())
cells(axes[2], fec_lo, fec_hi, sizes, 'FEC',
      r"\textbf{Eventi costanti}, 40 gruppi da circa %s eventi, di durata variabile" % f"{q:,}".replace(',', '\\,'))
cells(axes[3], vc_edges[:-1], vc_edges[1:], vc_counts, 'VC',
      r"\textbf{Lunghezza variabile}, %d bin da 164\,ms" % Tvc, lw=0.45)
for a in axes[1:]:
    a.tick_params(left=False)
axes[-1].spines['bottom'].set_visible(True)
axes[-1].set_xlabel("tempo dall'inizio della registrazione [s]")
axes[-1].set_xlim(0, WIN)
axes[-1].xaxis.set_major_locator(MultipleLocator(2))
axes[-1].xaxis.set_major_formatter(comma(0))
save(fig, 'cap4_tre_clock')
