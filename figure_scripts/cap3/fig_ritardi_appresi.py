"""Ritardi appresi nel mixer su DVS-Lip (sezione 3.5, paragrafo "Ritardi appresi").

Fonte: learned_delay_trajectory.csv del run E0_C0_M--f-tcap-stage1-dwc3-learnable-delays (seed 42, finestra fissa),
che registra a ogni epoca, per ciascun mixer e ramo, il centro medio del ritardo continuo sui canali e la frazione
di canali il cui ritardo arrotondato differisce da quello iniziale. Checkpoint selezionato: best_epoch del summary.
(a,b) spostamento medio rispetto al ritardo iniziale nei due patch embedding; (c) frazione di canali spostati al
checkpoint selezionato. Rampa ordinale blu 250/400/550/700 (validata con --ordinal).
"""
import csv
from common3 import *  # noqa

RUN = "E0_C0_M--f-tcap-stage1-dwc3-learnable-delays__20260914_112808_095061__seed42"
DIR = f"{RESEARCH}/artifacts/runs/dvslip/{RUN}"
MODULES = [("patch_embed1", "primo patch embedding"), ("patch_embed2", "secondo patch embedding")]
DELAYS = [1, 2, 4, 8]
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

rows = list(csv.DictReader(open(f"{DIR}/learned_delay_trajectory.csv")))
best = json.load(open(f"{DIR}/summary.json"))["best_epoch"]
data = {}
for r in rows:
    key = (r["module"].split(".")[0], int(r["branch"]))
    data.setdefault(key, []).append((int(r["epoch"]), float(r["mean_center"]) - float(r["initial_delay"]),
                                     float(r["fraction_hard_changed"])))
for k in data:
    data[k].sort()
for (m, _), _ in zip(MODULES, range(2)):
    for b, d in enumerate(DELAYS):
        e, s, c = zip(*data[(m, b)])
        i = e.index(best)
        print(f"{m} d={d}: shift {s[i]:+.3f}  changed {c[i]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), gridspec_kw=dict(width_ratios=[1, 1, 0.95], wspace=0.34))
for ax, (m, name), tag in zip(axes[:2], MODULES, "ab"):
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.text(3, 0.515, "mezzo passo", fontsize=6.3, color=MUTED, va="bottom")
    ax.axvline(best, color=MUTED, lw=0.6, zorder=1)
    for b, (d, col) in enumerate(zip(DELAYS, RAMP)):
        e, s, _ = zip(*data[(m, b)])
        ax.plot(e, s, color=col, lw=1.6, zorder=3, label=rf"${d}$")
    ax.set_xlim(0, 130)
    ax.set_xticks([0, 32, 64, 96, 128])
    ax.set_ylim(-0.08, 0.84)
    ax.yaxis.set_major_formatter(comma(1))
    ax.set_xlabel("epoca", fontsize=7.5)
    ax.set_title(rf"({tag}) {name}", loc="left", fontsize=8.5)
axes[0].set_ylabel(r"spostamento medio [passi]", fontsize=7.5)
axes[0].legend(title=r"ritardo iniziale [passi]", title_fontsize=6.6, fontsize=6.6, loc="upper left", ncol=4,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)

# (c) frazione di canali spostati al checkpoint selezionato
ax = axes[2]
w = 0.18
for gi, (m, _) in enumerate(MODULES):
    for b, (d, col) in enumerate(zip(DELAYS, RAMP)):
        e, _, c = zip(*data[(m, b)])
        v = 100 * c[e.index(best)]
        x = gi + (b - 1.5) * (w + 0.02)
        ax.bar(x, v, width=w, color=col, edgecolor="white", linewidth=0.6, zorder=2)
        if v >= 1:
            ax.text(x, v + 1.5, rf"${v:.0f}$", ha="center", va="bottom", fontsize=6.2, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["primo", "secondo"], fontsize=7)
ax.set_xlabel("patch embedding", fontsize=7.5)
ax.set_ylim(0, 88)
ax.set_xlim(-0.55, 1.55)
ax.yaxis.set_major_formatter(comma(0))
ax.set_ylabel(r"canali spostati [\%]", fontsize=7.5)
ax.grid(axis="x", visible=False)
ax.set_title(r"(c) al checkpoint selezionato", loc="left", fontsize=8.5)

save(fig, "cap3_ritardi_appresi")
