"""Stile comune delle figure della tesi (LaTeX, lmodern, palette validata)."""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = "/home/emanuele/Scrivania/Tesi/Emanuele_Bernacchi___Thesis___Spiking_Transformers/images/cap4"
WIDTH = 5.8  # pollici, circa \textwidth

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"

# Slot categorici 1-3 della palette validata (all-pairs PASS, light).
CLOCK = {
    "E0": dict(color="#2a78d6", marker="o", label="Finestra fissa"),
    "FEC": dict(color="#eb6834", marker="s", label="Eventi costanti"),
    "VC": dict(color="#1baf7a", marker="^", label="Lunghezza variabile"),
}
NEUTRAL = "#9a9993"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def comma(decimals=0, percent=False):
    """Formattatore con virgola decimale italiana."""
    def f(v, _pos=None):
        s = f"{v:.{decimals}f}".replace(".", "{,}")
        return s + (r"\%" if percent else "")
    return FuncFormatter(f)


def num(v, decimals=1):
    return f"{v:.{decimals}f}".replace(".", "{,}")


def save(fig, name):
    import os
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf")
    preview = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anteprime")
    os.makedirs(preview, exist_ok=True)
    fig.savefig(f"{preview}/{name}.png", dpi=170)
    plt.close(fig)
