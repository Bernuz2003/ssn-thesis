"""Base comune per le quattro bande (sezione 6.3): figura cap6_base_comune, schema.

(a) Le quattro matrici W_k (C x C), impilate, si approssimano con una base comune P (q x C) e quattro matrici D_k
    (C x q): W_k ~ D_k P, con P dai primi q vettori singolari destri della pila.
(b) Che cosa conserva la memoria del sito. Denso: la corrente z (C valori) per ciascuno degli ultimi d_K passi, medie di
    banda, W_k. Compresso: la proiezione Pz (q valori) calcolata una volta sola quando la corrente entra in memoria,
    medie di banda sulle proiezioni, D_k. Le altezze delle colonne sono proporzionali a C = 128 e q = 64 (DVS-Lip,
    modello compatto). Il ramo relazionale (A z, B z) non cambia ed e' omesso.
Verificato su src/etsr/models/interval.py e sui report di compressione (compressione6.py).
"""
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from common6 import *
from common5 import BAND

W, H = WIDTH, 3.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
BLUE = DSCOL['dvslip']


def box(x, y, w, h, s, fc="white", ec=INK2, fs=7.2, color=INK, lw=0.9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def rect(x0, y0, w, h, fc, ec=INK2, lw=0.7, s=None, fs=7, color=INK):
    ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=ec, lw=lw, zorder=3))
    if s:
        ax.text(x0 + w / 2, y0 + h / 2, s, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(p0, p1, col=INK2, lw=0.9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, shrinkA=0,
                                 shrinkB=0, zorder=2))


def node(x, y, s, r=0.075):
    ax.add_patch(Circle((x, y), r, fc="white", ec=INK2, lw=0.9, zorder=3))
    ax.text(x, y - 0.005, s, ha="center", va="center", fontsize=8, color=INK, zorder=4)


def label(x, y, s, fs=7, color=INK2, ha="center", va="center"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=5)


# ---- (a) la base comune --------------------------------------------------------------------------------------
label(0.0, H - 0.1, r"\textbf{(a)} una base comune per le quattro matrici", fs=8.3, color=INK, ha="left")
top, hk = 3.33, 0.24          # altezza di ciascun blocco della pila
xw, ww = 0.35, 0.62           # pila delle W_k (C colonne)
for k in range(4):
    rect(xw, top - (k + 1) * hk, ww, hk, tint(BAND[k], 0.35), s=rf"$W_{k + 1}$")
label(xw + ww / 2, top - 4 * hk - 0.13, r"quattro matrici $C\times C$", fs=6.6)
label(xw + ww + 0.2, top - 2 * hk, r"$\approx$", fs=11, color=INK)
xd, wd = 1.33, 0.31           # D_k (q colonne)
for k in range(4):
    rect(xd, top - (k + 1) * hk, wd, hk, tint(BAND[k], 0.35), s=rf"$D_{k + 1}$")
label(xd + wd / 2, top - 4 * hk - 0.13, r"$C\times q$", fs=6.6)
xp, wp, hp = 1.78, ww, 0.31 * hk / 0.24 * 0.5
rect(xp, top - hp, wp, hp, tint(BLUE, 0.85), ec=BLUE, s=r"$P$", color='white')
label(xp + wp / 2, top - hp - 0.13, r"base $q\times C$", fs=6.6)
xt = xp + wp + 0.25
label(xt, 3.17, r"$W_k\approx D_k\,P$", fs=7.8, color=INK, ha="left")
label(xt, 2.90, r"le righe di $P$ sono le $q$ direzioni dei canali", fs=6.8, ha="left")
label(xt, 2.73, r"che contano per tutte e quattro le bande", fs=6.8, ha="left")
label(xt, 2.48, r"ogni banda conserva una propria $D_k$,", fs=6.8, ha="left")
label(xt, 2.31, r"che riporta le $q$ componenti nei $C$ canali", fs=6.8, ha="left")

# ---- (b) che cosa conserva la memoria --------------------------------------------------------------------------
label(0.0, 2.08, r"\textbf{(b)} che cosa conserva la memoria in ogni posizione", fs=8.3, color=INK, ha="left")
C_H, Q_H = 0.56, 0.28           # altezze proporzionali a C = 128 e q = 64
NCOL, CW = 9, 0.085             # passi conservati (illustrativi) e larghezza di una colonna


def memory(x0, yc, h, col):
    for i in range(NCOL):
        a = 0.25 + 0.75 * i / (NCOL - 1)
        rect(x0 + i * (CW + 0.02), yc - h / 2, CW, h, tint(col, a), ec='white', lw=0.5)
    return x0 + NCOL * (CW + 0.02) - 0.02


def pipeline(yc, compressed):
    label(0.0, yc + 0.36, "IRM compresso" if compressed else "IRM denso", fs=7.3, color=INK, ha="left")
    box(0.28, yc, 0.4, 0.22, r"$z_t$", fc=INK, ec=INK, color="white")
    x = 0.48
    if compressed:
        arrow((x, yc), (0.62, yc))
        box(0.78, yc, 0.32, 0.22, r"$P$", fc=tint(BLUE, 0.85), ec=BLUE, color="white")
        x = 0.94
        label(0.78, yc - 0.26, "una volta\nper passo", fs=6.0)
    x0 = x + 0.18
    arrow((x, yc), (x0 - 0.04, yc))
    h = Q_H if compressed else C_H
    x1 = memory(x0, yc, h, BLUE)
    label((x0 + x1) / 2, yc - h / 2 - 0.12,
          (r"$q$ valori per passo" if compressed else r"$C$ valori per passo"), fs=6.6)
    label((x0 + x1) / 2, yc + h / 2 + 0.1, r"ultimi $d_K$ passi", fs=6.2)
    arrow((x1 + 0.04, yc), (x1 + 0.22, yc))
    xm = x1 + 0.64
    box(xm, yc, 0.8, 0.24, "medie di banda", fs=6.6)
    arrow((xm + 0.4, yc), (xm + 0.58, yc))
    xk = xm + 0.83
    box(xk, yc, 0.46, 0.24, r"$D_k$" if compressed else r"$W_k$", fc=tint(BAND[1], 0.35), ec=BAND[1])
    label(xk, yc - 0.26, r"$C\times q$" if compressed else r"$C\times C$", fs=6.4)
    arrow((xk + 0.23, yc), (xk + 0.48, yc))
    node(xk + 0.56, yc, r"$\Sigma$")
    arrow((xk + 0.64, yc), (xk + 0.86, yc))
    label(xk + 0.9, yc, r"richiamo", fs=6.8, ha="left")
    return x1


pipeline(1.50, compressed=False)
pipeline(0.46, compressed=True)
label(W - 0.05, 0.03, r"il ramo del confronto, con $Az_t$ e $Bz_t$, non cambia ed è omesso", fs=6.2, ha="right",
      va="bottom")
save(fig, 'cap6_base_comune')
