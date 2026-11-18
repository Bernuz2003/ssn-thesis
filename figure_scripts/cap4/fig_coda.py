import numpy as np
from style import *
from load import DS, load, macro_f1

names, tail, steps_share, sop_share, acc_full, acc_nt, f1_full, f1_nt = [], [], [], [], [], [], [], []
for d, C, name in DS:
    E = load(d, 'E0')
    tf = 1 - E['last'] / E['steps']
    names.append(name); tail.append(tf)
    steps_share.append(1 - E['nt_macs'].sum() / E['macs'].sum())
    sop_share.append(1 - E['nt_sops'].sum() / E['sops'].sum())
    acc_full.append(E['summary']['accuracy']); acc_nt.append(E['summary']['no_tail']['accuracy'])
    f1_full.append(macro_f1(E['pred'], E['targets'], C)); f1_nt.append(macro_f1(E['nt_pred'], E['targets'], C))
for n, a, b, c, d_, e, f in zip(names, steps_share, sop_share, acc_full, acc_nt, f1_full, f1_nt):
    print(f"{n:18s} steps_tail={a:.3f} sop_tail={b:.3f} acc {c:.4f}->{d_:.4f} f1 {e:.4f}->{f:.4f}")

# --- Figura unica: coda vuota (a), calcolo speso nella coda (b), dipendenza appresa dalla coda (c)
from matplotlib.lines import Line2D
fig = plt.figure(figsize=(WIDTH, 4.35))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.25, 1], wspace=0.12, hspace=0.62)
a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
y = np.arange(len(names))[::-1]
for yi, tf in zip(y, tail):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.22, 0.22, len(tf))
    a1.scatter(tf * 100, yi + jitter, s=2.2, color=CLOCK['E0']['color'], alpha=0.18, linewidths=0, rasterized=True)
    q1, med, q3 = np.percentile(tf * 100, [25, 50, 75])
    a1.plot([q1, q3], [yi, yi], color=INK, linewidth=2.2, solid_capstyle='round')
    a1.plot([med], [yi], marker='o', markersize=5.5, color='white', markeredgecolor=INK, markeredgewidth=1.2)
    a1.text(101, yi, r"$%s\%%$" % num(med, 0), va='center', ha='left', fontsize=8, color=INK)
a1.set_yticks(y); a1.set_yticklabels(names)
a1.set_xlim(0, 112); a1.set_xticks([0, 25, 50, 75, 100])
a1.xaxis.set_major_formatter(comma(0, percent=True))
a1.set_xlabel("passi della finestra dopo l'ultimo evento", fontsize=8)
a1.set_title(r"\textbf{(a)} coda vuota per campione", loc='left')
a1.grid(axis='y', visible=False)
a1.tick_params(axis='y', length=0)

h = 0.34
a2.barh(y + h / 2, np.array(steps_share) * 100, height=h, color=NEUTRAL, edgecolor='white', linewidth=0.8, label='passi e MAC densi')
a2.barh(y - h / 2, np.array(sop_share) * 100, height=h, color=CLOCK['E0']['color'], edgecolor='white', linewidth=0.8, label='operazioni sinaptiche')
for i, (yi, a, b) in enumerate(zip(y, steps_share, sop_share)):
    ta, tb = (r"\quad passi e MAC densi", r"\quad operazioni sinaptiche") if i == 0 else ("", "")
    a2.text(a * 100 + 1.2, yi + h / 2, r"$%s\%%$" % num(a * 100, 0) + ta, va='center', fontsize=7.5, color=INK2)
    a2.text(b * 100 + 1.2, yi - h / 2, r"$%s\%%$" % num(b * 100, 0) + tb, va='center', fontsize=7.5, color=INK)
a2.set_yticks(y); a2.set_yticklabels([]); a2.tick_params(axis='y', length=0)
a2.set_xlim(0, 92); a2.xaxis.set_major_formatter(comma(0, percent=True))
a2.set_xlabel("quota del totale spesa nella coda", fontsize=8)
a2.set_title(r"\textbf{(b)} calcolo speso nella coda", loc='left')
a2.grid(axis='y', visible=False)

for yi, n, a, b in zip(y, names, acc_full, acc_nt):
    a3.plot([b * 100, a * 100], [yi, yi], color=GRID, linewidth=3.0, solid_capstyle='round', zorder=1)
    a3.plot(a * 100, yi, marker='o', markersize=7, color=CLOCK['E0']['color'], markeredgecolor='white', markeredgewidth=1.0, zorder=3)
    a3.plot(b * 100, yi, marker='o', markersize=7, color='white', markeredgecolor=CLOCK['E0']['color'], markeredgewidth=1.4, zorder=3)
    a3.text(a * 100 + 1.3, yi, r"$%s$" % num(a * 100, 1), va='center', fontsize=8, color=INK)
    a3.text(b * 100 - 1.3, yi, r"$%s$" % num(b * 100, 1), va='center', ha='right', fontsize=8, color=INK2)
    a3.text(1.015, yi, r"$-%s$ punti" % num((a - b) * 100, 1), va='center', ha='left', fontsize=8, color=INK, transform=a3.get_yaxis_transform())
a3.set_yticks(y); a3.set_yticklabels(names); a3.tick_params(axis='y', length=0)
a3.set_xlim(25, 95); a3.xaxis.set_major_formatter(comma(0, percent=True))
a3.set_xlabel("accuratezza sulla validazione di sviluppo", fontsize=8)
a3.grid(axis='y', visible=False)
a3.set_title(r"\textbf{(c)} accuratezza con la coda e senza", loc='left')
a3.legend(handles=[Line2D([], [], marker='o', linestyle='', markersize=7, color=CLOCK['E0']['color'], markeredgecolor='white', label="sequenza completa"),
                   Line2D([], [], marker='o', linestyle='', markersize=7, color='white', markeredgecolor=CLOCK['E0']['color'], markeredgewidth=1.4, label="interrotta all'ultimo evento")],
          loc='lower right', bbox_to_anchor=(0.995, 0.02), fontsize=7, ncol=1, handletextpad=0.3, borderaxespad=0.1)
save(fig, 'cap4_coda')
