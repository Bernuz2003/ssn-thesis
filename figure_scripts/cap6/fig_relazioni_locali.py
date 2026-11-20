"""Relazioni locali prima del pooling (sezione 6.2): figura cap6_relazioni_locali, schema.

(a) IRM nel secondo patch embedding: in ogni posizione della mappa 8 x 8 il presente e' confrontato con il passato
    della stessa posizione, riassunto in quattro bande fino a 15 passi (411 ms su DVS-Lip).
(b) Relazioni locali nell'ultimo strato del frontend piramidale, sulla mappa 64 x 64 prima del pooling: la proiezione u
    di una cella p e' confrontata con quella di una cella vicina p + delta, lungo quattro direzioni, con ritardi di 1, 2 e
    4 passi (27-110 ms), con il prodotto speculare; le relazioni sono mediate su gruppi 3 x 3 con passo 2 e sommate alla
    corrente dopo il max pooling.
Verificato su src/etsr/models/local_prepool.py (R = u[t-d,p] u[t,p+delta] - u[t-d,p+delta] u[t,p]).
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from common6 import *
from common5 import BAND

W, H = WIDTH, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
BLUE = DSCOL['dvslip']


def rect(x0, y0, w, h, fc, ec=INK2, lw=0.6):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=ec, lw=lw, zorder=3))


def arrow(p0, p1, col=INK2, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=col, shrinkA=0,
                                 shrinkB=0, zorder=4))


def label(x, y, s, fs=7, color=INK2, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5)


def box(x, y, w, h, s, fs=6.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="white", ec=INK2, lw=0.9, zorder=3))
    label(x, y, s, fs=fs, color=INK)


# ---- (a) IRM ------------------------------------------------------------------------------------------------------
label(0.0, H - 0.1, r"\textbf{(a)} IRM, dopo il pooling", fs=8.3, color=INK, ha="left")
g0x, g0y, n, c = 0.12, 1.05, 8, 0.14
for i in range(n):
    for j in range(n):
        rect(g0x + i * c, g0y + j * c, c, c, "white", ec="#c9c9c9", lw=0.5)
rect(g0x + 4 * c, g0y + 4 * c, c, c, BLUE, ec=INK, lw=0.8)
label(g0x + n * c / 2, g0y - 0.14, r"mappa $8\times8$", fs=6.6)
tx0, ty, steps, sw = 1.55, 1.75, 16, 0.075
edges = [(1, 2), (3, 4), (5, 7), (8, 15)]
for s_ in range(steps):
    x = tx0 + (steps - 1 - s_) * sw
    k = next((i for i, (a_, b_) in enumerate(edges) if a_ <= s_ <= b_), None)
    fc = INK if s_ == 0 else (tint(BAND[k], 0.6) if k is not None else "white")
    rect(x, ty, sw * 0.85, 0.22, fc, ec="white", lw=0.4)
label(tx0 + (steps - 1) * sw + sw * 0.43, ty + 0.52, r"$t$", fs=7, color=INK)
label(tx0 + 0.4, ty + 0.34, "passato della stessa posizione", fs=6.2, ha="center")
for i, (a_, b_) in enumerate(edges):
    xa = tx0 + (steps - 1 - b_) * sw; xb = tx0 + (steps - 1 - a_) * sw + sw * 0.85
    ax.plot([xa, xb], [ty - 0.06, ty - 0.06], color=BAND[i], lw=1.4, zorder=3, solid_capstyle='butt')
label(tx0 + 0.6, ty - 0.24, r"quattro bande, fino a $411\,\mathrm{ms}$", fs=6.4)
arrow((g0x + 4.5 * c + 0.08, g0y + 4.5 * c), (tx0 - 0.06, ty + 0.11), col=INK2, lw=0.8)
label(2.15, 1.05, "confronto fra il presente\ne il passato riassunto\ndella stessa posizione", fs=6.6)

# ---- (b) relazioni locali ---------------------------------------------------------------------------------------
x0 = 3.1
label(x0, H - 0.1, r"\textbf{(b)} relazioni locali, prima del pooling", fs=8.3, color=INK, ha="left")
f0x, f0y, m, cf = x0 + 0.05, 1.12, 9, 0.125
for i in range(m):
    for j in range(m):
        rect(f0x + i * cf, f0y + j * cf, cf, cf, "white", ec="#d6d6d6", lw=0.4)
pc = (4, 5)
rect(f0x + pc[0] * cf, f0y + pc[1] * cf, cf, cf, BLUE, ec=INK, lw=0.8)
for (dx, dy) in [(1, 0), (0, -1), (1, -1), (-1, -1)]:
    rect(f0x + (pc[0] + dx) * cf, f0y + (pc[1] + dy) * cf, cf, cf, tint(BLUE, 0.35), ec=BLUE, lw=0.7)
    cx = f0x + (pc[0] + 0.5) * cf; cy = f0y + (pc[1] + 0.5) * cf
    arrow((cx, cy), (cx + dx * cf * 0.95, cy + dy * cf * 0.95), col=INK, lw=0.7, ms=5)
label(f0x + m * cf / 2, f0y - 0.2, "mappa $64\\times64$, cella $p$\ne quattro vicine $p+\\delta$", fs=6.4)
rx, ry = 4.55, 2.0
for s_ in range(5):
    x = rx + (4 - s_) * 0.2
    fc = INK if s_ == 0 else (tint(BLUE, 0.55) if s_ in (1, 2, 4) else "white")
    rect(x, ry, 0.17, 0.2, fc, ec=INK2 if s_ == 3 else "white", lw=0.4)
for s_, lab in [(0, r"$t$"), (1, r"$t{-}1$"), (2, r"$t{-}2$"), (4, r"$t{-}4$")]:
    label(rx + (4 - s_) * 0.2 + 0.085, ry + 0.31, lab, fs=6.2, color=INK)
label(rx + 0.5, ry - 0.15, r"ritardi da $27$ a $110\,\mathrm{ms}$", fs=6.4)
box(4.45, 0.62, 2.6, 0.3, r"$R_{d,\delta,t}(p)=u_{t-d}(p)\odot u_t(p{+}\delta)-u_{t-d}(p{+}\delta)\odot u_t(p)$", fs=6.8)
arrow((f0x + m * cf / 2, f0y - 0.4), (f0x + m * cf / 2, 0.79))
arrow((rx + 0.5, ry - 0.27), (rx + 0.5, 0.79))
label(4.45, 0.2, "media su gruppi $3\\times3$ con passo $2$,\npoi sommata alla corrente dopo il max-pooling", fs=6.4)
save(fig, 'cap6_relazioni_locali')
