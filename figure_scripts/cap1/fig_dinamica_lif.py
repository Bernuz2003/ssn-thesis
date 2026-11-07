"""Integrazione, soglia e soft reset in un neurone IF e in un neurone LIF a tempo discreto.

Stesso ingresso totale (tre impulsi da 0,6) consegnato in passi consecutivi oppure distanziati.
Aggiornamento H_k = lam * V_{k-1} + X_k, soglia V_th = 1, soft reset V_k = H_k - S_k V_th.
IF: lam = 1, LIF: lam = 0,5. Valori riportati nel testo della sezione 1.2.1.
"""
import numpy as np
from common1 import *  # noqa

IF_C = "#eb6834"
LIF_C = "#2a78d6"
VTH = 1.0
K = 10
AMP = 0.6


def simulate(times, lam):
    x = np.zeros(K + 1)
    x[list(times)] = AMP
    v = 0.0
    path_k, path_v, spikes, peaks = [0], [0.0], [], []
    for k in range(1, K + 1):
        h = lam * v + x[k]
        path_k.append(k)
        path_v.append(h)
        peaks.append(h)
        if h >= VTH:
            spikes.append(k)
            v = h - VTH
            path_k.append(k)
            path_v.append(v)
        else:
            v = h
    return x, np.array(path_k), np.array(path_v), spikes, peaks


cases = [
    ((1, 2, 3), r"(a) tre ingressi in passi consecutivi"),
    ((1, 4, 7), r"(b) gli stessi ingressi, distanziati"),
]

fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 3.15), sharex=True,
                         gridspec_kw=dict(height_ratios=[0.55, 3.0, 1], hspace=0.10, wspace=0.12))

for col, (times, title) in enumerate(cases):
    axs, ax, axi = axes[0, col], axes[1, col], axes[2, col]
    x = None
    for lam, color, ls, lab, mk in [(1.0, IF_C, (0, (4, 2)), r"IF", "s"),
                                    (0.5, LIF_C, "-", r"LIF, $\lambda=0{,}5$", "o")]:
        x, pk, pv, spikes, peaks = simulate(times, lam)
        ax.plot(pk, pv, color=color, ls=ls, lw=1.4, label=lab, zorder=3)
        # marker sul potenziale appena caricato H_k
        hk = np.arange(1, K + 1)
        ax.plot(hk, peaks, ls="none", marker=mk, ms=3.6, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        yspk = 1 if lam == 1.0 else 0
        for s in spikes:
            axs.plot([s], [yspk], marker="|", ms=7, mew=2.0, color=color, zorder=5)
        print(title, lab, "H =", [round(p, 3) for p in peaks], "spike ai passi", spikes)

    ax.axhline(VTH, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=2)
    ax.text(0.05, VTH + 0.04, r"$V_{\mathrm{th}}$", va="bottom", ha="left", fontsize=8, color=INK2)
    ax.set_ylim(-0.05, 1.32)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(comma(1))
    axs.set_title(title, loc="left")
    axs.set_ylim(-0.7, 1.7)
    axs.set_yticks([0, 1])
    axs.grid(False)
    axs.spines["left"].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    axs.tick_params(bottom=False, left=False)
    if col == 0:
        ax.set_ylabel(r"potenziale")
        axs.set_yticklabels([r"LIF", r"IF"])
        axs.set_ylabel(r"spike", rotation=0, ha="right", va="center", labelpad=14)
    else:
        ax.tick_params(labelleft=False)
        axs.tick_params(labelleft=False)

    axi.vlines(np.arange(K + 1), 0, x, color=INK, lw=1.6)
    axi.plot(np.arange(K + 1)[x > 0], x[x > 0], "o", ms=3, color=INK)
    axi.set_ylim(0, 0.85)
    axi.set_yticks([0, 0.6])
    axi.yaxis.set_major_formatter(comma(1))
    axi.set_xlim(-0.3, K + 0.3)
    axi.set_xticks(range(0, K + 1))
    axi.set_xlabel(r"passo $k$")
    if col == 0:
        axi.set_ylabel(r"$X_k$")
    else:
        axi.tick_params(labelleft=False)
    axi.grid(axis="x", visible=False)

axes[1, 0].annotate(r"$1{,}05$", xy=(3, 1.05), xytext=(4.0, 1.13), fontsize=7.5, color=LIF_C,
                    arrowprops=dict(arrowstyle="-", color=LIF_C, lw=0.6))
axes[1, 1].annotate(r"massimo $0{,}68$", xy=(7, 0.684), xytext=(7.7, 0.52), fontsize=7.5,
                    color=LIF_C, arrowprops=dict(arrowstyle="-", color=LIF_C, lw=0.6))
axes[1, 0].legend(loc="center right", bbox_to_anchor=(1.0, 0.42), handlelength=2.2, fontsize=7.5)

save(fig, "cap1_dinamica_lif")
