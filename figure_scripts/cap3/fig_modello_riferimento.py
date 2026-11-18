"""Il modello di riferimento e le tre modifiche rispetto alla baseline (sezione 3.2).

Verificato su src/etsr/models/layers.py (PyramidalPatchEmbedding, ConvBNLIF2d, PatchEmbeddingStage,
SpikingDepthwiseLocalMixer) e sulle configurazioni del run di riferimento:
frontend piramidale con conv 3x3 2->8 a piena risoluzione, poi conv+BN+max-pooling+LIF verso 16, 32, 64
canali, scorciatoia 1x1 con passo 4 dalla seconda mappa; mixer a ritardi [1, 2, 4, 8] sulle correnti,
prima del LIF, nell'ultimo strato del primo patch embedding e nella convoluzione con passo del secondo;
stadio 1 con convoluzione depthwise 3x3 al posto della Q-K Token Attention.
"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from common3 import *  # noqa

W, H = WIDTH, 3.25
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, lines, col, mod=False, fs=7.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=tint(col, 0.16), ec=INK if mod else col, lw=1.5 if mod else 0.9, zorder=3))
    n = len(lines)
    for i, s in enumerate(lines):
        ax.text(x, y + (n - 1) * 0.075 - i * 0.15, s, ha="center", va="center", fontsize=fs, color=INK, zorder=4)


def arrow(x0, y0, x1, y1, col=INK2, lw=0.9, conn=None, ms=7):
    kw = dict(connectionstyle=conn) if conn else {}
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col,
                                 shrinkA=0, shrinkB=0, zorder=2, **kw))


def label(x, y, s, fs=7, color=INK2, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5)


# (a) sequenza dei blocchi
label(0.0, H - 0.07, r"(a) modello di riferimento, con le modifiche rispetto alla baseline in evidenza",
      fs=8.5, color=INK, ha="left")
y, bh = 2.62, 0.52
blocks = [
    (0.37, 0.64, ["ingresso"], HEAD, False, r"$2\times128\times128$"),
    (1.29, 0.80, ["patch emb. 1", "piramidale"], PE1, True, r"$64\times16\times16$"),
    (2.19, 0.66, ["stadio 1", "depthwise"], ST1, True, r"$64\times16\times16$"),
    (3.08, 0.80, ["patch", "embedding 2"], PE2, False, r"$128\times8\times8$"),
    (3.97, 0.66, ["stadio 2", "SSA"], ST2, False, r"$128\times8\times8$"),
    (4.83, 0.68, ["readout", "media"], HEAD, False, r"$128$"),
    (5.52, 0.50, ["testa"], HEAD, False, r"$C$"),
]
for i, (x, w, lines, col, mod, out) in enumerate(blocks):
    box(x, y, w, bh, lines, col, mod)
    label(x, y - bh / 2 - 0.12, out)
    if i:
        xp, wp = blocks[i - 1][0], blocks[i - 1][1]
        arrow(xp + wp / 2 + 0.02, y, x - w / 2 - 0.02, y)
for x in (1.29, 3.08):   # mixer a ritardi nei due patch embedding
    ax.add_patch(FancyBboxPatch((x - 0.27, y + bh / 2 + 0.05), 0.54, 0.17, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=INK, ec=INK, lw=0.6, zorder=4))
    label(x, y + bh / 2 + 0.135, r"+ ritardi", fs=6.5, color="white")

# (b) frontend piramidale
label(0.0, 2.02, r"(b) frontend piramidale", fs=8.5, color=INK, ha="left")
yb = 1.22
chain = [(0.45, 0.42, 128, 8), (1.10, 0.31, 64, 16), (1.66, 0.22, 32, 32), (2.12, 0.15, 16, 64)]
for i, (x, s, res, ch) in enumerate(chain):
    ax.add_patch(Rectangle((x - s / 2, yb - s / 2), s, s, fc=tint(PE1, 0.30), ec=PE1, lw=0.8, zorder=3))
    label(x, yb - 0.36, rf"${res}\times{res}$", fs=6.8)
    label(x, yb - 0.49, rf"{ch} canali", fs=6.8)
    if i:
        xp, sp = chain[i - 1][0], chain[i - 1][1]
        arrow(xp + sp / 2 + 0.02, yb, x - s / 2 - 0.02, yb, col=PE1)
label(0.78, yb + 0.08, r"pool", fs=6)
label(1.385, yb + 0.08, r"pool", fs=6)
label(1.89, yb + 0.08, r"pool", fs=6)
xs = 2.55
ax.add_patch(Circle((xs, yb), 0.065, fc="white", ec=INK2, lw=0.8, zorder=4))
label(xs, yb - 0.004, r"$+$", fs=8, color=INK)
arrow(chain[-1][0] + chain[-1][1] / 2 + 0.02, yb, xs - 0.065, yb, col=PE1)
x1 = chain[1][0]
ax.plot([x1, x1, xs], [yb + chain[1][1] / 2, yb + 0.36, yb + 0.36], color=PE1, lw=0.9, zorder=2)
arrow(xs, yb + 0.36, xs, yb + 0.065, col=PE1)
label((x1 + xs) / 2, yb + 0.44, r"scorciatoia $1\times1$, passo 4", fs=6.8)
ax.add_patch(FancyBboxPatch((chain[3][0] - 0.2, yb + 0.12), 0.4, 0.13, boxstyle="round,pad=0,rounding_size=0.02",
                            fc=INK, ec=INK, lw=0.5, zorder=4))
label(chain[3][0], yb + 0.185, r"ritardi", fs=5.8, color="white")
label(1.30, 0.38, r"baseline\,: 32 canali a piena risoluzione".replace("\\,:", ","), fs=6.8)

# (c) mixer a ritardi
label(3.0, 2.02, r"(c) mixer causale a più ritardi", fs=8.5, color=INK, ha="left")
yc = 0.95
xs0, dx = 3.12, 0.29
X = [xs0 + i * dx for i in range(9)]           # da t-8 a t
for i, x in enumerate(X):
    d = 8 - i
    sel = d in (0, 1, 2, 4, 8)
    ax.add_patch(Rectangle((x - 0.09, yc - 0.09), 0.18, 0.18, fc=tint(PE2, 0.55 if d else 1.0) if sel else "#e6e5e1",
                           ec=PE2 if sel else "#c9c8c3", lw=0.7, zorder=3))
    label(x, yc - 0.22, r"$z_t$" if d == 0 else rf"$z_{{t-{d}}}$", fs=6.3)
xsum, ysum = X[-1], yc + 0.62
ax.add_patch(Circle((xsum, ysum), 0.07, fc="white", ec=INK2, lw=0.8, zorder=4))
label(xsum, ysum - 0.004, r"$+$", fs=8, color=INK)
arrow(X[-1], yc + 0.09, xsum, ysum - 0.07, col=INK2)
for d, rad in [(1, -0.35), (2, -0.32), (4, -0.28), (8, -0.22)]:
    x = X[8 - d]
    arrow(x, yc + 0.09, xsum - 0.06, ysum + 0.02, col=PE2, lw=0.9, conn=f"arc3,rad={rad}")
    label(x - 0.02, yc + 0.20 + 0.035 * d, rf"$W_{{{d}}}$", fs=6.3, color=INK)
box(xsum, ysum + 0.42, 0.42, 0.22, ["LIF"], HEAD, fs=7)
arrow(xsum, ysum + 0.07, xsum, ysum + 0.31, col=INK2)
label(xsum - 0.24, ysum + 0.19, r"$u_t$", fs=7, color=INK, ha="right")
label(4.33, 0.38, r"$u_t=z_t+\sum_k W_k\,z_{t-d_k}$, con $d_k\in\{1,2,4,8\}$", fs=7)

save(fig, "cap3_modello_riferimento")
