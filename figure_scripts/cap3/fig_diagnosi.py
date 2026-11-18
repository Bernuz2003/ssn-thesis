"""Diagnosi della baseline su DVS-Lip (sezione 3.1).

(a) quota delle MAC e degli accumuli potenziali per blocco, dal profilo comune della baseline;
(b) Macro-F1 della baseline e delle varianti diagnostiche, seed 42, validazione di sviluppo.
"""
from common3 import *  # noqa
from matplotlib.patches import Rectangle

p = profile(CHAIN[0][1])
mac = block_totals(p, "multivalued_mac_potential")
ac = block_totals(p, "binary_ac_potential")
tot_mac, tot_ac = sum(mac.values()), sum(ac.values())
print("MAC tot G", tot_mac / 1e9, "AC tot G", tot_ac / 1e9)
for k, n, _ in BLOCKS:
    print(f"  {n:18s} MAC {100 * mac[k] / tot_mac:5.1f}%  AC {100 * ac[k] / tot_ac:5.1f}%")
pe = (mac["patch_embed1"] + mac["patch_embed2"] + ac["patch_embed1"] + ac["patch_embed2"]) / (tot_mac + tot_ac)
print(f"  patch embedding su tutte le operazioni potenziali: {100 * pe:.1f}%")

variants = [
    ("baseline", f1(summary_glob("dvslip", "E0_B_M"))),
    ("senza memoria tra i passi", f1(summary_glob("dvslip", "E0_B_M--no-cross-time"))),
    ("media fino all'ultimo evento", f1(summary_glob("dvslip", "E0_B_M--readout-time-last-event"))),
    (r"larghezza 192 ($1{,}1$ M parametri)", f1(summary_glob("dvslip", "E0_B_M--capacity-1m"))),
    (r"larghezza 256 ($2{,}0$ M parametri)", f1(summary_glob("dvslip", "E0_B_M--capacity-2m"))),
]
for n, v in variants:
    print(f"  {n:40s} {v:.2f}")

fig = plt.figure(figsize=(WIDTH, 3.25))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.62)

# (a) quote per blocco
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100); ax.set_ylim(-0.6, 1.6)
ax.axis("off")
ax.set_title(r"(a) dove si concentrano le operazioni potenziali", loc="left", fontsize=8.5)
for y, vals, tot, lab in [(1.0, mac, tot_mac, r"MAC"), (0.0, ac, tot_ac, r"accumuli potenziali")]:
    ax.text(-1.5, y, lab, ha="right", va="center", fontsize=7.5, color=INK)
    x = 0.0
    for k, n, c in BLOCKS:
        w = 100 * vals[k] / tot
        if w <= 0.05:
            continue
        ax.add_patch(Rectangle((x, y - 0.32), max(w - 0.35, 0.05), 0.64, fc=c, ec="none"))
        if w > 6:
            ax.text(x + w / 2, y, rf"${w:.0f}\%$", ha="center", va="center", fontsize=7, color=INK)
        x += w
    ax.text(101.5, y, rf"${tot / 1e9:.2f}$ G".replace(".", "{,}"), ha="left", va="center", fontsize=7, color=INK2)
xl = 0
for k, n, c in BLOCKS[:4]:
    ax.add_patch(Rectangle((xl, -0.62), 1.6, 0.26, fc=c, ec="none"))
    ax.text(xl + 2.4, -0.49, n, ha="left", va="center", fontsize=6.8, color=INK2)
    xl += 2.4 + 1.25 * len(n) + 4

# (b) varianti diagnostiche
ax = fig.add_subplot(gs[1])
names = [v[0] for v in variants][::-1]
vals = [v[1] for v in variants][::-1]
y = np.arange(len(vals))
cols = [NEUTRAL if i != len(vals) - 1 else INK2 for i in range(len(vals))]
ax.barh(y, vals, height=0.58, color=[tint(ST1, 0.55)] * (len(vals) - 1) + [ST1], edgecolor="none")
for yi, v in zip(y, vals):
    ax.text(v + 0.8, yi, rf"${v:.1f}$".replace(".", "{,}"), va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlim(0, 60)
ax.xaxis.set_major_formatter(comma(0))
ax.set_xlabel(r"Macro-F1 sulla validazione di sviluppo [\%]", fontsize=7.5)
ax.grid(axis="y", visible=False)
ax.axvline(vals[-1], color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=0)
ax.set_title(r"(b) baseline e varianti diagnostiche su DVS-Lip", loc="left", fontsize=8.5)

save(fig, "cap3_diagnosi")
