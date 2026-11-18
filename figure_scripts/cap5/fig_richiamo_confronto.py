"""Dal richiamo al confronto (figura 5.1): mixer a ritardi, correlatore di Hassenstein-Reichardt, IRM.

Schema verificato su src/etsr/models/interval.py: medie di banda sulla storia disponibile, richiamo W_k m_k,
relazione (A m)(B z) - (B m)(A z) proiettata da U_k, entrambi moltiplicati dal coefficiente b_{k,t}, sommati sulle
bande e alla corrente presente z_t.
"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from common5 import *

W, H = WIDTH, 3.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ANTI = "#2a78d6"


def box(x, y, w, h, s, fc="white", ec=INK2, fs=7.2, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def node(x, y, s, r=0.075):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK2, lw=0.9, zorder=3))
    ax.text(x, y - 0.005, s, ha="center", va="center", fontsize=8, color=INK, zorder=4)


def arrow(p0, p1, col=INK2, lw=0.9, conn=None):
    kw = dict(connectionstyle=conn) if conn else {}
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, shrinkA=0,
                                 shrinkB=0, zorder=2, **kw))


def line(xs, ys, col=INK2, lw=0.9):
    ax.plot(xs, ys, color=col, lw=lw, zorder=2, solid_capstyle="round")


def label(x, y, s, fs=7, color=INK2, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5)


# (a) mixer a ritardi
label(0.0, H - 0.08, r"\textbf{(a)} mixer a ritardi: il passato viene richiamato", fs=8.3, color=INK, ha="left")
ys = [3.38, 3.08, 2.78, 2.48]
for k, y in enumerate(ys, 1):
    box(0.42, y, 0.62, 0.21, rf"$z_{{t-d_{k}}}$", fc=tint(BAND[k - 1], 0.25), ec=BAND[k - 1])
    box(1.15, y, 0.42, 0.21, rf"$W_{k}$")
    arrow((0.74, y), (0.93, y))
    arrow((1.37, y), (1.86, 2.93), conn="arc3,rad=0")
node(1.93, 2.93, r"$+$")
box(1.93, 3.42, 0.42, 0.21, r"$z_t$", fc=INK, ec=INK, color="white")
arrow((1.93, 3.31), (1.93, 3.005))
arrow((2.005, 2.93), (2.45, 2.93))
label(2.55, 2.93, r"$u_t$", fs=8, color=INK, ha="left")
label(1.3, 2.18, "ogni termine dipende da un solo istante", fs=6.8)

# (b) correlatore di Hassenstein e Reichardt
x0 = 3.05
label(x0, H - 0.08, r"\textbf{(b)} correlatore di Hassenstein e Reichardt", fs=8.3, color=INK, ha="left")
sx1, sx2, ys_ = 3.55, 5.15, 3.42
for sx, s in ((sx1, r"$s_1$"), (sx2, r"$s_2$")):
    ax.add_patch(Circle((sx, ys_), 0.11, fc=tint(ANTI, 0.2), ec=ANTI, lw=0.9, zorder=3))
    label(sx, ys_, s, fs=7.5, color=INK)
box(sx1, 3.02, 0.34, 0.2, r"$\delta$"); box(sx2, 3.02, 0.34, 0.2, r"$\delta$")
arrow((sx1, ys_ - 0.11), (sx1, 3.12)); arrow((sx2, ys_ - 0.11), (sx2, 3.12))
mx1, mx2, my = 3.98, 4.72, 2.62
node(mx1, my, r"$\times$"); node(mx2, my, r"$\times$")
arrow((sx1, 2.92), (mx1 - 0.06, my + 0.05))          # s1 ritardato
arrow((sx2 - 0.08, ys_ - 0.08), (mx1 + 0.05, my + 0.06))  # s2 istantaneo
arrow((sx2, 2.92), (mx2 + 0.06, my + 0.05))          # s2 ritardato
arrow((sx1 + 0.08, ys_ - 0.08), (mx2 - 0.05, my + 0.06))  # s1 istantaneo
node(4.35, 2.27, r"$-$")
arrow((mx1 + 0.04, my - 0.07), (4.30, 2.33)); arrow((mx2 - 0.04, my - 0.07), (4.40, 2.33))
arrow((4.425, 2.27), (4.85, 2.27))
label(4.93, 2.27, r"$R(t)$", fs=8, color=INK, ha="left")
label(3.2, 2.62, "ritardato\n$\\times$ istantaneo", fs=6.5)

# (c) IRM
label(0.0, 1.93, r"\textbf{(c)} IRM: il passato viene richiamato e confrontato con il presente", fs=8.3, color=INK, ha="left")
yt, ym, yl = 1.5, 0.98, 0.42
edges = [(0.12, 0.96), (0.98, 1.38), (1.40, 1.65), (1.67, 1.86)]
for k, (a, b) in enumerate(edges):
    kk = 3 - k
    ax.add_patch(Rectangle((a, yt - 0.11), b - a, 0.22, fc=BAND[kk], ec="white", lw=1.0, zorder=3))
    label((a + b) / 2, yt + 0.2, rf"$I_{kk + 1}$", fs=6.8)
box(2.1, yt, 0.36, 0.22, r"$z_t$", fc=INK, ec=INK, color="white")
box(1.0, ym, 0.55, 0.24, r"$m_{k,t}$", fs=7.5)
label(0.36, ym, "media sulla\nstoria disponibile", fs=6.3)
arrow((1.0, yt - 0.11), (1.0, ym + 0.12))
box(2.1, ym, 0.42, 0.22, r"$W_k$")
arrow((1.275, ym), (1.89, ym))
label(1.68, ym + 0.17, "richiamo", fs=6.6)
box(3.0, yl, 1.95, 0.3, r"$(A m_{k,t})\odot(B z_t)-(B m_{k,t})\odot(A z_t)$", fc=tint(ANTI, 0.14), ec=ANTI, fs=6.9)
arrow((1.0, ym - 0.12), (2.025, yl), conn="angle,angleA=-90,angleB=180,rad=0")
line([2.28, 2.6], [yt, yt]); arrow((2.6, yt), (2.6, yl + 0.15))
label(3.0, yl - 0.27, "confronto antisimmetrico", fs=6.6, color=ANTI)
box(4.27, yl, 0.42, 0.22, r"$U_k$")
arrow((3.975, yl), (4.06, yl))
node(4.27, ym, r"$+$")
arrow((2.31, ym), (4.195, ym)); arrow((4.27, yl + 0.11), (4.27, ym - 0.075))
box(4.8, ym, 0.48, 0.22, r"$\times\,b_{k,t}$")
arrow((4.345, ym), (4.56, ym))
node(5.3, ym, r"$+$")
arrow((5.04, ym), (5.225, ym))
line([2.6, 5.3], [yt, yt]); arrow((5.3, yt), (5.3, ym + 0.075))
label(5.3, ym - 0.19, r"$\sum\nolimits_k$", fs=7)
arrow((5.375, ym), (5.62, ym))
label(5.66, ym, r"$u_t$", fs=8, color=INK, ha="left")
save(fig, "cap5_richiamo_confronto")
