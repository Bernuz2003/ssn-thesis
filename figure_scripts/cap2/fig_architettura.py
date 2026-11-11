"""Architettura della baseline compatta (sezione 2.3).

Struttura verificata su src/etsr/models/{mini_qkformer,layers}.py (commit 5a03fec):
patch embedding 1 = conv 3x3 2->32 (passo 1), 32->64, 64->64, 64->64 (passo 2), scorciatoia 1x1 passo 8
dalla prima mappa; stadio 1 = QKTA (8 teste) + MLP 64->128->64; patch embedding 2 = conv 3x3 64->128,
conv 3x3 passo 2, scorciatoia 1x1 passo 2; stadio 2 = SSA (8 teste) + MLP 128->256->128; readout = media
su spazio e tempo; testa lineare. Parametri e stato per blocco: tabella 2.3 (DVS-Lip, C = 100).
Palette: slot 4, 1, 3, 2 della palette validata (ordine adiacente verificato), testa in grigio neutro.
"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from common2 import *  # noqa

PE1, ST1, PE2, ST2, HEAD = "#eda100", "#2a78d6", "#1baf7a", "#eb6834", "#9a9993"


def tint(color, a):
    r = np.array(to_rgb(color))
    return tuple(1 - a * (1 - r))


W, H = WIDTH, 3.62
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, lines, col, fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=tint(col, 0.16), ec=col, lw=0.9, zorder=3))
    n = len(lines)
    for i, s in enumerate(lines):
        ax.text(x, y + (n - 1) * 0.075 - i * 0.15, s, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(x0, y0, x1, y1, col=INK2, lw=0.9, conn=None):
    kw = dict(connectionstyle=conn) if conn else {}
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=col,
                                 shrinkA=0, shrinkB=0, zorder=2, **kw))


def label(x, y, s, fs=7, color=INK2, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5)


# (a) struttura della rete
label(0.0, H - 0.12, r"(a) struttura della rete", fs=8.5, color=INK, ha="left")
y, bh = 3.02, 0.52
blocks = [  # x, w, righe, colore, uscita
    (0.37, 0.64, ["ingresso"], HEAD, r"$2\times128\times128$"),
    (1.29, 0.80, ["patch", "embedding 1"], PE1, r"$64\times16\times16$"),
    (2.19, 0.66, ["stadio 1", "QKTA"], ST1, r"$64\times16\times16$"),
    (3.08, 0.80, ["patch", "embedding 2"], PE2, r"$128\times8\times8$"),
    (3.97, 0.66, ["stadio 2", "SSA"], ST2, r"$128\times8\times8$"),
    (4.83, 0.68, ["readout", "media"], HEAD, r"$128$"),
    (5.52, 0.50, ["testa"], HEAD, r"$C$"),
]
for i, (x, w, lines, col, out) in enumerate(blocks):
    box(x, y, w, bh, lines, col if i not in (0, 5, 6) else HEAD)
    label(x, y - bh / 2 - 0.12, out)
    if i:
        xp, wp = blocks[i - 1][0], blocks[i - 1][1]
        arrow(xp + wp / 2 + 0.02, y, x - w / 2 - 0.02, y)

# (b) dettaglio del primo patch embedding
label(0.0, 2.47, r"(b) primo patch embedding", fs=8.5, color=INK, ha="left")
yb = 1.80
chain = [(0.72, 0.46, 128, 32), (1.50, 0.34, 64, 64), (2.18, 0.25, 32, 64), (2.76, 0.18, 16, 64)]
for i, (x, s, res, ch) in enumerate(chain):
    ax.add_patch(Rectangle((x - s / 2, yb - s / 2), s, s, fc=tint(PE1, 0.30), ec=PE1, lw=0.8, zorder=3))
    label(x, yb - 0.38, rf"${res}\times{res}$")
    label(x, yb - 0.52, rf"{ch} canali")
    if i:
        xp, sp = chain[i - 1][0], chain[i - 1][1]
        arrow(xp + sp / 2 + 0.02, yb, x - s / 2 - 0.02, yb, col=PE1)
xs = 3.32
ax.add_patch(Circle((xs, yb), 0.07, fc="white", ec=INK2, lw=0.8, zorder=4))
label(xs, yb - 0.004, r"$+$", fs=8, color=INK)
arrow(chain[-1][0] + chain[-1][1] / 2 + 0.02, yb, xs - 0.07, yb, col=PE1)
# scorciatoia dalla prima mappa
x0 = chain[0][0]
ax.plot([x0, x0, xs], [yb + chain[0][1] / 2, yb + 0.36, yb + 0.36], color=PE1, lw=0.9, zorder=2)
arrow(xs, yb + 0.36, xs, yb + 0.07, col=PE1)
label((x0 + xs) / 2, yb + 0.44, r"scorciatoia $1\times1$ con passo 8")
label(1.11, yb + 0.10, r"passo 2", fs=6.5)
arrow(xs + 0.07, yb, 3.72, yb)
label(3.78, yb + 0.07, r"$64\times16\times16$", ha="left")
label(3.78, yb - 0.09, r"valori $\{0,1,2\}$", ha="left")
label(3.78, yb - 0.27, r"unico modulo a piena risoluzione", ha="left")

# (c) distribuzione di parametri e stato
label(0.0, 0.98, r"(c) distribuzione su DVS-Lip", fs=8.5, color=INK, ha="left")
names = ["patch embedding 1", "stadio 1", "patch embedding 2", "stadio 2", "testa"]
cols = [PE1, ST1, PE2, ST2, HEAD]
params = np.array([95_360, 29_440, 230_144, 132_864, 12_900])
state = np.array([884_736, 100_352, 49_152, 65_536, 0])
X0, BW, BH = 0.95, 4.20, 0.17
for yy, vals, lab, tot in [(0.70, params, "parametri", r"$500\,708$"), (0.42, state, "stato", r"$1\,099\,776$")]:
    label(X0 - 0.08, yy, lab, fs=7.5, color=INK, ha="right")
    x = X0
    for v, c in zip(vals, cols):
        w = BW * v / vals.sum()
        if w <= 0:
            continue
        ax.add_patch(Rectangle((x, yy - BH / 2), max(w - 0.02, 0.005), BH, fc=c, ec="none", zorder=3))
        if w > 0.32:
            label(x + w / 2, yy, rf"${100 * v / vals.sum():.0f}\%$", fs=7, color=INK)
        x += w
    label(X0 + BW + 0.07, yy, tot, fs=7, ha="left")
# legenda
xl = X0
for n, c in zip(names, cols):
    ax.add_patch(Rectangle((xl, 0.10), 0.09, 0.09, fc=c, ec="none", zorder=3))
    label(xl + 0.13, 0.145, n, fs=6.8, ha="left")
    xl += 0.13 + 0.058 * len(n) + 0.16

save(fig, "cap2_architettura")
print("quote parametri", np.round(100 * params / params.sum(), 1), "stato", np.round(100 * state / state.sum(), 1))
