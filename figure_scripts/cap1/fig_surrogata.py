"""Surrogate gradient: gradino nel passaggio in avanti, derivata sigmoidale nel passaggio all'indietro.

g_alpha(z) = alpha * sigma(alpha z) * (1 - sigma(alpha z)), con z = H - V_th.
alpha = 4 e' il valore adottato nella baseline (capitolo 2); 2 e 8 per confronto.
"""
import numpy as np
from common1 import *  # noqa
from matplotlib.ticker import FuncFormatter

RAMP = {2: "#86b6ef", 4: "#2a78d6", 8: "#104281"}   # rampa ordinale blu, validata
z = np.linspace(-2, 2, 801)


def sig(u):
    return 1.0 / (1.0 + np.exp(-u))


fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.25), gridspec_kw=dict(wspace=0.28))

# (a) passaggio in avanti
a1.plot(z[z < 0], np.zeros((z < 0).sum()), color=INK, lw=1.8)
a1.plot(z[z >= 0], np.ones((z >= 0).sum()), color=INK, lw=1.8, label=r"$\Theta(z)$")
a1.plot([0, 0], [0, 1], color=INK, lw=0.8, ls=(0, (1, 1.5)))
a1.plot(z, sig(4 * z), color=RAMP[4], lw=1.4, ls=(0, (4, 2)),
        label=r"$\sigma(4z)$")
a1.set_title(r"(a) passaggio in avanti", loc="left")
a1.set_xlabel(r"$z=H_k-V_{\mathrm{th}}$")
a1.set_ylabel(r"spike $S_k$")
a1.set_ylim(-0.08, 1.25)
a1.set_yticks([0, 1])
a1.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=7.5, handlelength=2.2)

# (b) passaggio all'indietro
a2.axhline(0, color=INK, lw=1.8, zorder=2)
a2.annotate("", xy=(0, 2.25), xytext=(0, 0), zorder=3,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=7))
a2.text(0.3, 2.17, r"$\Theta'$ nulla ovunque,", fontsize=7.5, color=INK, va="center")
a2.text(0.3, 1.98, r"impulso in $z=0$", fontsize=7.5, color=INK, va="center")
for alpha, c in RAMP.items():
    g = alpha * sig(alpha * z) * (1 - sig(alpha * z))
    a2.plot(z, g, color=c, lw=2.0 if alpha == 4 else 1.3, zorder=4, label=rf"$\alpha={alpha}$")
a2.set_title(r"(b) passaggio all'indietro", loc="left")
a2.set_xlabel(r"$z=H_k-V_{\mathrm{th}}$")
a2.set_ylabel(r"derivata")
a2.set_ylim(-0.08, 2.35)
a2.yaxis.set_major_formatter(comma(1))
a2.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=7.5, handlelength=1.6)

for ax in (a1, a2):
    ax.set_xlim(-2, 2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: rf"${v:.0f}$"))

for a in (2, 4, 8):
    print("alpha", a, "picco", a / 4,
          "g(0.5)", round(a * sig(a * 0.5) * (1 - sig(a * 0.5)), 3),
          "g(1)", round(a * sig(a) * (1 - sig(a)), 3))

save(fig, "cap1_surrogata")
