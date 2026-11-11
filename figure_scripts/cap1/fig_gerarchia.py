"""Struttura a scala singola (Spikformer) e struttura gerarchica (QKFormer), sezione 1.3.4.

Dimensioni dagli articoli originali.
Spikformer: SPS produce T x N x D, poi L blocchi SSA+MLP a dimensioni costanti, GAP e classificatore.
QKFormer: SPEDS-1 con patch 4x4, poi ogni SPEDS dimezza il lato e raddoppia i canali
          (T x H/4 x W/4 x C, T x H/8 x W/8 x 2C, T x H/16 x W/16 x 4C).
          Configurazione predefinita: QKTA nei primi stage, SSA nell'ultimo.
Altezza dei rettangoli proporzionale al lato della griglia dei token, larghezza ai canali.
"""
from schemi import *  # noqa

W, H = WIDTH, 2.72
fig, ax = canvas(W, H)
SC = "#eb6834"   # blocchi con SSA
QK = "#2a78d6"   # blocchi con Q-K Attention
NEU = tint(REAL, 0.20)
BH = 0.56


def blocco(x, y, w, testo, testo2=None, fc=NEU, ec=REAL):
    box(ax, x, y, w, BH, fc=fc, ec=ec)
    if testo2:
        label(ax, x, y + 0.08, testo, fs=7.5)
        label(ax, x, y - 0.09, testo2, fs=7.5)
    else:
        label(ax, x, y, testo, fs=7.5)


def feat(x, y, h, w, col):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=tint(col, 0.30), ec=col, lw=0.8, zorder=3))


# (a) Spikformer
y = 2.08
label(ax, 0.0, H - 0.11, r"(a) Spikformer, scala unica", fs=8.5, ha="left")
blocco(0.33, y, 0.56, "ingresso")
arrow(ax, 0.61, y, 0.80, y)
blocco(1.09, y, 0.58, "SPS")
xs = [1.92, 3.03, 4.14]
arrow(ax, 1.38, y, xs[0] - 0.19, y)
for i, x in enumerate(xs):
    feat(x, y, 0.38, 0.38, SC)
    if i < 2:
        arrow(ax, x + 0.19, y, xs[i + 1] - 0.19, y)
        testo = r"SSA + MLP" if i == 0 else r"$\cdots$"
        label(ax, (x + xs[i + 1]) / 2, y + 0.10, testo)
label(ax, xs[0], y - 0.31, r"$N$ token, $D$ canali", fs=7, color=INK2)
label(ax, xs[1], y - 0.31, r"$N$ token, $D$ canali", fs=7, color=INK2)
label(ax, xs[2], y - 0.31, r"dopo $L$ blocchi", fs=7, color=INK2)
arrow(ax, xs[2] + 0.19, y, 4.98, y)
blocco(5.31, y, 0.66, "classificatore")

# (b) QKFormer
y = 0.64
label(ax, 0.0, 1.40, r"(b) QKFormer, struttura gerarchica", fs=8.5, ha="left")
blocco(0.33, y, 0.56, "ingresso")
arrow(ax, 0.61, y, 0.74, y)
PX = [1.03, 2.21, 3.46]
FX = [1.66, 2.90, 4.40]
FH = [0.72, 0.36, 0.18]
FW = [0.14, 0.28, 0.56]
TXT = [r"$\frac{H}{4}{\times}\frac{W}{4}$, $C$", r"$\frac{H}{8}{\times}\frac{W}{8}$, $2C$",
       r"$\frac{H}{16}{\times}\frac{W}{16}$, $4C$"]
ATT = [r"QKTA", r"QKTA", r"SSA"]
COL = [QK, QK, SC]
for i in range(3):
    blocco(PX[i], y, 0.56, rf"SPEDS-{i + 1}")
    arrow(ax, PX[i] + 0.28, y, FX[i] - FW[i] / 2, y)
    feat(FX[i], y, FH[i], FW[i], COL[i])
    label(ax, FX[i], y + 0.48, ATT[i] + rf" $\times N_{i + 1}$", fs=7.5)
    label(ax, FX[i], y - 0.53, TXT[i], fs=7.5, color=INK2)
    if i < 2:
        arrow(ax, FX[i] + FW[i] / 2, y, PX[i + 1] - 0.28, y)
arrow(ax, FX[2] + FW[2] / 2, y, 4.98, y)
blocco(5.31, y, 0.66, "classificatore")

save(fig, "cap1_gerarchia")
