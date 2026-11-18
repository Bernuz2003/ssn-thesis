"""Solidità del modello di riferimento (sezione 3.3).

(a) baseline e modello di riferimento sui quattro dataset: barre = addestramento con lo stesso seed (42) per i due
    modelli, punti = ripetizioni ulteriori disponibili (DVS-Lip: seed 43 e 44 per entrambi; riferimento anche su
    Gesture seed 43, CIFAR10-DVS seed 43 e 44, THU-EACT-50-CHL seed 45);
(b) Macro-F1 in funzione dei parametri su DVS-Lip per le larghezze 128, 192 e 256 (seed 42).
Validazione di sviluppo, rappresentazione a finestra fissa.
"""
from common3 import *  # noqa

DATASETS = [("dvslip", r"DVS-Lip"), ("dvsgesture", r"DVS128" "\n" r"Gesture"),
            ("cifar10dvs", r"CIFAR10-" "\n" r"DVS"), ("thu_eact_50_chl", r"THU-EACT-" "\n" r"50-CHL")]
EXTRA = {"dvslip": (43, 44), "dvsgesture": (43,), "cifar10dvs": (43, 44), "thu_eact_50_chl": (45,)}
BASE_C, REF_C = tint(HEAD, 0.75), ST1

b42, r42, rex, bex = [], [], [], []
for ds, _ in DATASETS:
    b42.append(f1(summary_glob(ds, "E0_B_M")))
    r42.append(f1(summary_glob(ds, "E0_C0_M")))
    rex.append([f1(summary_glob(ds, "E0_C0_M", seed=s)) for s in EXTRA[ds]])
    bex.append([f1(summary_glob(ds, "E0_B_M", seed=s)) for s in (43, 44)] if ds == "dvslip" else [])
for (ds, _), b, r, re_, be in zip(DATASETS, b42, r42, rex, bex):
    print(f"{ds:16s} B {b:.2f} {[round(x, 2) for x in be]}  rif {r:.2f} {[round(x, 2) for x in re_]}  delta {r - b:+.2f}")

cap = {}
for arch in ("B", "C0"):
    pts = []
    for suffix in ("", "--capacity-1m", "--capacity-2m"):
        s = summary_glob("dvslip", f"E0_{arch}_M{suffix}")
        pts.append((s["trainable_parameters"] / 1e6, f1(s)))
    cap[arch] = pts
    print(arch, [(round(p, 3), round(v, 2)) for p, v in pts])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.65), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))

# (a) quattro dataset
x = np.arange(len(DATASETS))
w = 0.36
a1.bar(x - w / 2, b42, width=w, color=BASE_C, edgecolor="white", linewidth=0.8, label="baseline compatta", zorder=2)
a1.bar(x + w / 2, r42, width=w, color=REF_C, edgecolor="white", linewidth=0.8, label="modello di riferimento",
       zorder=2)
for i in range(len(DATASETS)):
    for v in bex[i]:
        a1.plot(x[i] - w / 2, v, "o", ms=2.6, color=INK, zorder=4)
    for v in rex[i]:
        a1.plot(x[i] + w / 2, v, "o", ms=2.6, color=INK, zorder=4)
    top = max([r42[i]] + rex[i])
    a1.text(x[i] + w / 2, top + 1.6, rf"$+{r42[i] - b42[i]:.1f}$".replace(".", "{,}"), ha="center", va="bottom",
            fontsize=7, color=INK)
    a1.text(x[i] - w / 2, 3, rf"${b42[i]:.1f}$".replace(".", "{,}"), ha="center", va="bottom", fontsize=6.3,
            color=INK, rotation=90)
    a1.text(x[i] + w / 2, 3, rf"${r42[i]:.1f}$".replace(".", "{,}"), ha="center", va="bottom", fontsize=6.3,
            color="white", rotation=90)
a1.set_xticks(x); a1.set_xticklabels([d[1] for d in DATASETS], fontsize=6.6)
a1.set_ylim(0, 112); a1.set_yticks(range(0, 101, 20)); a1.yaxis.set_major_formatter(comma(0))
a1.set_ylabel(r"Macro-F1 [\%]", fontsize=7.5)
a1.grid(axis="x", visible=False)
a1.legend(loc="upper left", fontsize=6.8, handlelength=1.2, borderaxespad=0.1, ncol=2, columnspacing=1.2)
a1.set_title(r"(a) quattro dataset", loc="left", fontsize=8.5)

# (b) capacità su DVS-Lip
for arch, col, name in (("B", INK2, "baseline compatta"), ("C0", REF_C, "modello di riferimento")):
    px, py = zip(*cap[arch])
    a2.plot(px, py, color=col, marker="o", ms=4, lw=1.6, zorder=3)
    for p, v in cap[arch]:
        dy = 1.3 if arch == "C0" else -1.3
        a2.text(p, v + dy, rf"${v:.1f}$".replace(".", "{,}"), ha="center", va="bottom" if dy > 0 else "top",
                fontsize=6.5, color=INK)
    a2.text(px[-1] + 0.1, py[-1], name.replace(" ", "\n", 1), va="center", ha="left", fontsize=6.6, color=INK2)
a2.axhline(cap["C0"][0][1], color=REF_C, lw=0.7, ls=(0, (2, 2)), zorder=1)
a2.set_xscale("log")
a2.set_xticks([0.5, 1.0, 2.0]); a2.set_xticklabels([r"$0{,}5$", r"$1$", r"$2$"])
a2.minorticks_off()
a2.set_xlim(0.42, 3.3)
a2.set_ylim(40, 66)
a2.yaxis.set_major_formatter(comma(0))
a2.set_xlabel(r"parametri [milioni]", fontsize=7.5)
a2.set_ylabel(r"Macro-F1 [\%]", fontsize=7.5)
a2.set_title(r"(b) capacità su DVS-Lip", loc="left", fontsize=8.5)

save(fig, "cap3_solidita")
