"""Quattro modi di suddividere lo stesso flusso sintetico (illustrativo)."""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from common1 import *

rng = np.random.default_rng(3)
segments = [(0.0, 0.35, 2000), (0.35, 0.55, 15000), (0.55, 1.17, 4000)]
t = np.sort(np.concatenate([rng.uniform(a, b, rng.poisson(r * (b - a))) for a, b, r in segments]))
N, tl = len(t), t[-1]
print('N', N, 'ultimo evento', tl)

rows = []
# 1. finestra fissa: 100 ms, 2 s
e = np.arange(21) * 0.1; rows.append(("Finestra fissa", r"$\Delta t=100$\,ms, $T=20$", e[:-1], e[1:], np.histogram(t, e)[0]))
# 2. registrazione intera, durata fissa
T2 = int(np.floor((tl - t[0]) / 0.1)) + 1; e = t[0] + np.arange(T2 + 1) * 0.1
rows.append(("Registrazione intera", r"$\Delta t=100$\,ms, $T=%d$" % T2, e[:-1], e[1:], np.histogram(t, e)[0]))
# 3. numero di passi fisso
T3 = 8; e = np.linspace(t[0], tl, T3 + 1); e[-1] += 1e-9
rows.append(("Numero di passi fisso", r"$T=8$, $\Delta t_n=%d$\,ms" % round((tl - t[0]) / T3 * 1000), e[:-1], e[1:], np.histogram(t, e)[0]))
# 4. eventi costanti
q, r = divmod(N, T3); sizes = np.array([q + 1] * r + [q] * (T3 - r)); st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
lo = t[st]; hi = np.append(t[st[1:]], tl)
rows.append(("Eventi costanti", r"$T=8$, circa %d eventi per passo" % q, lo, hi, sizes))

cmap = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#9ec5f4', '#2a78d6', '#0d366b'])
norm = Normalize(0, max(r[4].max() for r in rows))
fig, axes = plt.subplots(5, 1, figsize=(WIDTH, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 0.6, 0.6, 0.6, 0.6], hspace=0.45))
ax = axes[0]
re = np.arange(0, 2.0001, 0.01)
ax.fill_between(re[:-1], np.histogram(t, re)[0] / 10, step='post', color=NEUTRAL, alpha=0.6, linewidth=0)
ax.axvline(tl, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
ax.text(tl + 0.03, 6, "ultimo evento", fontsize=7.5, color=INK2, va='top')
ax.set_ylabel('eventi/ms'); ax.set_ylim(0, 22)
ax.text(0.45, 19.5, 'raffica', ha='center', fontsize=7.5, color=INK2)
for a, (name, detail, lo, hi, cnt) in zip(axes[1:], rows):
    for x0, x1, n in zip(lo, hi, cnt):
        a.add_patch(Rectangle((x0, 0), x1 - x0, 1, facecolor=cmap(norm(n)) if n > 0 else 'white',
                              edgecolor='white' if n > 0 else MUTED, linewidth=0.9 if n > 0 else 0.5,
                              hatch=None if n > 0 else '////'))
    a.set_ylim(0, 1); a.set_yticks([]); a.grid(False)
    for s_ in ('left',): a.spines[s_].set_visible(False)
    a.set_title(r"\textbf{%s}, %s" % (name, detail), loc='left', fontsize=8, pad=2)
from matplotlib.ticker import MultipleLocator
axes[-1].set_xlim(0, 2.0); axes[-1].set_xlabel('tempo [s]'); axes[-1].xaxis.set_major_locator(MultipleLocator(0.2)); axes[-1].xaxis.set_major_formatter(comma(1))
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = axes[0].inset_axes([0.66, 0.62, 0.3, 0.1])
cb = fig.colorbar(sm, cax=cax, orientation='horizontal'); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6.5, length=2); cb.ax.set_title('eventi per passo', fontsize=7, pad=2)
save(fig, 'cap1_partizioni')
