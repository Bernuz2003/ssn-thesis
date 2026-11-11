"""Primitive per gli schemi a blocchi della sezione 1.3.

Le figure sono disegnate su un asse che copre tutta la tela, con coordinate in pollici,
cosi' le dimensioni dei caratteri e delle forme restano prevedibili.
"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from common1 import *  # noqa

BIN = "#2a78d6"    # valori binari (spike)
INT = "#eb6834"    # valori interi o comunque a piu' bit
REAL = "#8a8984"   # valori reali
SURF = "#ffffff"


def tint(color, a):
    """Mescola il colore con il bianco, con a=1 pieno e a=0 bianco."""
    r = np.array(to_rgb(color))
    return tuple(1 - a * (1 - r))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=SURF, ec=INK2, lw=0.8, fs=8, color=INK, z=3):
    """Blocco arrotondato centrato in (x, y)."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.035",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=z + 1)


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=0.9, ls="-", head=True, z=2, conn=None):
    style = "-|>" if head else "-"
    kw = dict(connectionstyle=conn) if conn else {}
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=lw, color=color,
                        ls=ls, shrinkA=0, shrinkB=0, zorder=z, **kw)
    ax.add_patch(a)


def label(ax, x, y, text, fs=7.5, color=INK, ha="center", va="center", z=5):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, zorder=z)


def tensor(ax, x, y, values, kind, cw, ch, outline=False, vmax=None):
    """Disegna una matrice cella per cella, centrata in (x, y).

    kind = 'bin' (0/1 in blu), 'int' (intensita' di arancio), 'real' (intensita' di grigio).
    """
    values = np.atleast_2d(values).astype(float)
    n, d = values.shape
    x0 = x - d * cw / 2
    y0 = y + n * ch / 2
    vmax = vmax if vmax is not None else max(values.max(), 1e-9)
    base = dict(bin=BIN, int=INT, real=REAL)[kind]
    for i in range(n):
        for j in range(d):
            v = values[i, j] / vmax
            if kind == "bin":
                fc = tint(base, 1.0) if v > 0 else tint(base, 0.10)
            else:
                fc = tint(base, 0.10 + 0.90 * v)
            ax.add_patch(Rectangle((x0 + j * cw, y0 - (i + 1) * ch), cw, ch, fc=fc, ec=SURF, lw=0.35, zorder=3))
    w, h = d * cw, n * ch
    ax.add_patch(Rectangle((x0, y0 - h), w, h, fc="none", ec=INK if outline else tint(base, 0.75),
                           lw=1.4 if outline else 0.5, zorder=4))
    return w, h
