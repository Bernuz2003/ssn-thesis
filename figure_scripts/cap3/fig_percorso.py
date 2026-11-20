"""Dalla baseline al modello di riferimento su DVS-Lip (sezione 3.2): prestazioni e costi passo per passo.

Macro-F1 dalla validazione di sviluppo (seed 42); costi dai profili comuni (stessi 64 campioni, FP32, CPU).
MAC = moltiplicazioni con operando non binario, comprese quelle del mixer a ritardi; accumuli potenziali; stato
persistente; stima aritmetica ad attività (Horowitz, 45 nm), che non è energia misurata.
"""
from common3 import *  # noqa

LABELS = ["baseline", "frontend\npiramidale", "mixer a\nritardi", "conv.\ndepthwise"]
rows = []
for key, rid in CHAIN:
    s = summary("dvslip", rid); p = profile(rid); ops = p["operations_per_sample"]
    rows.append(dict(
        f1=f1(s), params=s["trainable_parameters"],
        mac=ops["multivalued_mac_potential"],  # comprende gia' le MAC del mixer a ritardi
        ac=ops["binary_ac_potential"], state=p["state"]["persistent_state_elements"],
        energy=p["energy_reference"]["covered_arithmetic_activity_proxy_uj_per_sample"] / 1e3))
for k, r in zip([c[0] for c in CHAIN], rows):
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.55), gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.42))
x = np.arange(len(rows))

# (a) Macro-F1
v = [r["f1"] for r in rows]
a1.bar(x, v, width=0.6, color=[tint(ST1, 0.45)] * 3 + [ST1], edgecolor="none")
for i, val in enumerate(v):
    a1.text(i, val + 0.8, rf"${val:.1f}$".replace(".", "{,}"), ha="center", va="bottom", fontsize=7, color=INK)
    if i:
        a1.text(i, 3, rf"$+{val - v[i - 1]:.1f}$".replace(".", "{,}"), ha="center", va="bottom", fontsize=6.8,
                color=INK)
a1.set_xticks(x); a1.set_xticklabels(LABELS, fontsize=6.6)
a1.set_ylim(0, 62); a1.yaxis.set_major_formatter(comma(0))
a1.set_ylabel(r"Macro-F1 [\%]", fontsize=7.5)
a1.grid(axis="x", visible=False)
a1.set_title(r"(a) prestazioni", loc="left", fontsize=8.5)

# (b) costi relativi alla baseline: tabella a colori, una riga per metrica e una colonna per passo (cumulativo)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
METRICS = [("params", "parametri"), ("mac", "MAC"), ("energy", "energia\nstimata"), ("ac", "accumuli\npotenziali"),
           ("state", "stato")]
R = np.array([[100 * rows[i][k] / rows[0][k] for i in range(len(rows))] for k, _ in METRICS])
for (k, name), r in zip(METRICS, R):
    print(f"  {name.replace(chr(10), ' '):20s}", [round(q, 1) for q in r])
cmap = LinearSegmentedColormap.from_list("div", [(0, "#0d3d78"), (0.25, ST1), (0.43, "#cfe0f6"), (0.5, "#f4f3ef"),
                                                (0.62, "#f6c9a8"), (1, ST2)])
norm = TwoSlopeNorm(vmin=40, vcenter=100, vmax=115)
a2.imshow(R, cmap=cmap, norm=norm, aspect="auto")
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        v = R[i, j]
        a2.text(j, i, rf"${v:.0f}\,\%$", ha="center", va="center", fontsize=7,
                color="white" if v < 62 else INK)
a2.set_xticks(x); a2.set_xticklabels(LABELS, fontsize=6.6)
a2.set_yticks(range(len(METRICS))); a2.set_yticklabels([n for _, n in METRICS], fontsize=6.8)
a2.tick_params(length=0)
for sp in a2.spines.values():
    sp.set_visible(False)
a2.set_xticks(np.arange(-0.5, len(rows), 1), minor=True); a2.set_yticks(np.arange(-0.5, len(METRICS), 1), minor=True)
a2.grid(which="minor", color="white", lw=1.5); a2.grid(which="major", visible=False)
a2.tick_params(which="minor", length=0)
a2.set_title(r"(b) costi rispetto alla baseline", loc="left", fontsize=8.5)

save(fig, "cap3_percorso")
