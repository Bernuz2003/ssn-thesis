"""Connessioni residue SEW e MS (sezione 1.3.3).

Formule di Spikingformer (Zhou et al., AAAI 2026), equazioni 4 e 5:
  SEW  O_l = SN_l(ConvBN_l(O_{l-1})) + O_{l-1}
  MS   O_l = ConvBN_l(SN_l(O_{l-1})) + O_{l-1}
"""
from schemi import *  # noqa

W, H = WIDTH, 2.16
fig, ax = canvas(W, H)
BW_C, BW_S, BH, R = 0.56, 0.34, 0.28, 0.085
ARC = 0.31


def somma(x, y):
    ax.add_patch(Circle((x, y), R, fc=SURF, ec=INK2, lw=0.8, zorder=4))
    label(ax, x, y - 0.005, r"$+$", fs=8)


def tratto(x0, x1, y, col, testo=None, head=True, dy=0.09, xt=None):
    arrow(ax, x0, y, x1, y, color=col, lw=1.3, head=head)
    if testo:
        label(ax, xt if xt is not None else (x0 + x1) / 2, y + dy, testo, fs=7)


def salto(x0, x1, y, col, testo=None):
    """Ramo residuo: sale in x0, corre in alto e scende nel nodo somma in x1."""
    arrow(ax, x0, y, x0, y + ARC, color=col, lw=1.1, head=False)
    arrow(ax, x0, y + ARC, x1, y + ARC, color=col, lw=1.1, head=False)
    arrow(ax, x1, y + ARC, x1, y + R, color=col, lw=1.1)
    if testo:
        label(ax, (x0 + x1) / 2, y + ARC + 0.09, testo, fs=7)


# (a) SEW: la somma avviene tra spike e i valori crescono
y = 1.42
label(ax, 0.0, H - 0.12, r"(a) residuo SEW, somma tra spike", fs=8.5, ha="left")
xs = [0.10, 0.92, 1.62, 2.14, 3.06, 3.74, 4.26, 5.72]
tratto(xs[0], xs[1] - BW_C / 2, y, BIN, r"$\{0,1\}$", xt=0.26)
box(ax, xs[1], y, BW_C, BH, "ConvBN", fc=tint(BIN, 0.12), ec=BIN, fs=7.5)
tratto(xs[1] + BW_C / 2, xs[2] - BW_S / 2, y, REAL)
box(ax, xs[2], y, BW_S, BH, "SN", fs=7.5)
tratto(xs[2] + BW_S / 2, xs[3] - R, y, BIN)
somma(xs[3], y)
salto(0.54, xs[3], y, BIN, r"$\{0,1\}$")
tratto(xs[3] + R, xs[4] - BW_C / 2, y, INT, r"$\{0,1,2\}$", xt=2.52)
box(ax, xs[4], y, BW_C, BH, "ConvBN", fc=tint(INT, 0.18), ec=INT, fs=7.5)
tratto(xs[4] + BW_C / 2, xs[5] - BW_S / 2, y, REAL)
box(ax, xs[5], y, BW_S, BH, "SN", fs=7.5)
tratto(xs[5] + BW_S / 2, xs[6] - R, y, BIN)
somma(xs[6], y)
salto(xs[3] + 0.135, xs[6], y, INT, r"$\{0,1,2\}$")
tratto(xs[6] + R, xs[7], y, INT, r"$\{0,1,2,3\}$")
label(ax, xs[4], y - 0.25, r"pesi moltiplicati per un intero", fs=7, color=INK2)

# (b) MS: la somma avviene tra potenziali, gli strati pesati ricevono solo spike
y = 0.36
label(ax, 0.0, 0.97, r"(b) residuo MS, somma tra potenziali di membrana", fs=8.5, ha="left")
xs = [0.10, 0.70, 1.50, 2.14, 2.98, 3.78, 4.42, 5.72]
tratto(xs[0], xs[1] - BW_S / 2, y, REAL, r"$\mathbb{R}$", xt=0.40)
box(ax, xs[1], y, BW_S, BH, "SN", fs=7.5)
tratto(xs[1] + BW_S / 2, xs[2] - BW_C / 2, y, BIN, r"$\{0,1\}$")
box(ax, xs[2], y, BW_C, BH, "ConvBN", fc=tint(BIN, 0.12), ec=BIN, fs=7.5)
tratto(xs[2] + BW_C / 2, xs[3] - R, y, REAL)
somma(xs[3], y)
salto(0.24, xs[3], y, REAL, r"potenziale")
tratto(xs[3] + R, xs[4] - BW_S / 2, y, REAL, r"$\mathbb{R}$", xt=2.62)
box(ax, xs[4], y, BW_S, BH, "SN", fs=7.5)
tratto(xs[4] + BW_S / 2, xs[5] - BW_C / 2, y, BIN, r"$\{0,1\}$")
box(ax, xs[5], y, BW_C, BH, "ConvBN", fc=tint(BIN, 0.12), ec=BIN, fs=7.5)
tratto(xs[5] + BW_C / 2, xs[6] - R, y, REAL)
somma(xs[6], y)
salto(xs[3] + 0.16, xs[6], y, REAL, r"potenziale")
tratto(xs[6] + R, xs[7], y, REAL, r"$\mathbb{R}$")
label(ax, xs[5], y - 0.25, r"pesi solo accumulati", fs=7, color=INK2)

save(fig, "cap1_residui")
