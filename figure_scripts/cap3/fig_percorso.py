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

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.55), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.32))
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

# (b) costi relativi alla baseline
series = [("params", "parametri", INK2, (0, (3, 2))), ("mac", "MAC", ST2, "-"), ("ac", "accumuli potenziali", ST1, "-"),
          ("state", "stato", PE2, "-"), ("energy", "stima energetica", PE1, "-")]
ends = []
for key, name, col, ls in series:
    rel = [100 * r[key] / rows[0][key] for r in rows]
    a2.plot(x, rel, color=col, ls=ls, lw=1.5, marker="o", ms=3.5, zorder=3)
    ends.append((rel[-1], name, col))
    print(f"  {name:20s}", [round(q, 1) for q in rel])
# etichette dirette a destra, distanziate
ends.sort()
ylab = []
for val, name, col in ends:
    yy = val if not ylab or val - ylab[-1] > 7 else ylab[-1] + 7
    ylab.append(yy)
    a2.text(len(rows) - 1 + 0.12, yy, name, va="center", fontsize=6.8, color=INK2)
a2.axhline(100, color=GRID, lw=0.8, zorder=0)
a2.set_xticks(x); a2.set_xticklabels(LABELS, fontsize=6.6)
a2.set_xlim(-0.25, len(rows) - 1 + 1.25)
a2.set_ylim(30, 130)
a2.yaxis.set_major_formatter(comma(0, percent=True))
a2.set_ylabel(r"rispetto alla baseline", fontsize=7.5)
a2.grid(axis="x", visible=False)
a2.set_title(r"(b) costi", loc="left", fontsize=8.5)

save(fig, "cap3_percorso")
