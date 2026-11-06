import numpy as np
from matplotlib.lines import Line2D
from style import *
from load import DS, load

# Macro-F1 (%) dalle summary di addestramento, C0 Mean, validazione di sviluppo.
# (seed 42 di riferimento, altre repliche indipendenti)
R = {
 'DVS-Lip':        {'E0': (55.18, [54.88, 56.56]), 'FEC': (47.23, []), 'VC': (56.85, [])},
 'DVS128 Gesture': {'E0': (86.52, [86.52]),        'FEC': (94.83, []), 'VC': (94.23, [95.19])},
 'CIFAR10-DVS':    {'E0': (67.79, [67.24, 68.08]), 'FEC': (70.31, []), 'VC': (69.36, [67.17, 68.27])},
 'THU-EACT-50-CHL':{'E0': (45.55, [48.67]),        'FEC': None,        'VC': (54.93, [])},
}
fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.15), gridspec_kw=dict(wspace=0.55))
for ax, (name, row) in zip(axes, R.items()):
    vals = []
    for i, key in enumerate(('E0', 'FEC', 'VC')):
        c = CLOCK[key]
        if row[key] is None:
            ax.text(i, 0.5, "n.d.", transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=8, color=MUTED)
            continue
        ref, reps = row[key]; vals += [ref] + reps
        for k, rv in enumerate(reps):
            ax.plot(i + 0.17 * (k + 1), rv, marker=c['marker'], markersize=4.5, color='white', markeredgecolor=c['color'], markeredgewidth=1.1, linestyle='')
        ax.plot(i, ref, marker=c['marker'], markersize=7, color=c['color'], markeredgecolor='white', markeredgewidth=0.8, linestyle='')
        ax.text(i - 0.17, ref, r"$%s$" % num(ref, 1), ha='right', va='center', fontsize=7.2, color=INK)
    lo, hi = min(vals), max(vals); pad = max(1.2, (hi - lo) * 0.18)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.75, 2.65)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['F', 'E', 'L'])
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_formatter(comma(0))
    ax.set_title(name, fontsize=8.5)
axes[0].set_ylabel(r"Macro-F1 [\%]")
handles = [Line2D([], [], marker=CLOCK[k]['marker'], linestyle='', color=CLOCK[k]['color'], markersize=6.5, label=f"{l} {CLOCK[k]['label'].lower()}") for k, l in (('E0', 'F'), ('FEC', 'E'), ('VC', 'L'))]
handles.append(Line2D([], [], marker='o', linestyle='', color='white', markeredgecolor=INK2, markersize=4.5, label='altri seed'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4, handletextpad=0.2, columnspacing=1.2)
save(fig, 'cap4_risultati_clock')

# --- Accuratezza per quartile di durata, finestra fissa contro lunghezza variabile
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.2), gridspec_kw=dict(wspace=0.5))
for ax, (d, C, name) in zip(axes, DS):
    E = load(d, 'E0'); V = load(d, 'VC')
    q = np.quantile(E['duration'], [.25, .5, .75]); b = np.digitize(E['duration'], q)
    meds = [np.median(E['duration'][b == i]) / 1e6 for i in range(4)]
    for key, M, off in (('E0', E, -0.08), ('VC', V, 0.08)):
        c = CLOCK[key]; acc = []
        for i in range(4):
            m = b == i; k, n = M['correct'][m].sum(), m.sum(); lo_, hi_ = wilson(k, n)
            ax.plot([i + off, i + off], [lo_ * 100, hi_ * 100], color=c['color'], linewidth=1.0, alpha=0.45)
            acc.append(k / n * 100)
        ax.plot(np.arange(4) + off, acc, color=c['color'], marker=c['marker'], markersize=4.5, linewidth=1.4, markeredgecolor='white', markeredgewidth=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels([r"$%s$" % num(v, 2 if v < 2 else 1) for v in meds], fontsize=7)
    ax.set_xlim(-0.4, 3.4)
    ax.yaxis.set_major_formatter(comma(0))
    ax.set_title(name, fontsize=8.5)
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel(r"accuratezza [\%]")
fig.text(0.5, -0.04, "durata mediana del quartile [s]", ha='center', fontsize=9)
handles = [Line2D([], [], marker=CLOCK[k]['marker'], color=CLOCK[k]['color'], markersize=5, label=CLOCK[k]['label']) for k in ('E0', 'VC')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.5)
save(fig, 'cap4_quartili_durata')

# --- Settling e costante di tempo su Gesture
fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.25), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.95))
tt = np.linspace(0, 3, 601)
for lab, dt, tau, col, ls in (("finestra fissa, $\\Delta t=500$\\,ms, $\\tau_d=2$", 0.5, 2.0, CLOCK['E0']['color'], '-'),
                              ("lunghezza variabile, $\\Delta t=164$\\,ms, $\\tau_d=2$", 0.164358, 2.0, CLOCK['VC']['color'], '-'),
                              ("lunghezza variabile, $\\tau_d\\approx4{,}91$", 0.164358, 4.90785, INK2, (0, (3, 2)))):
    lam = 1 - 1 / tau
    a1.plot(tt, lam ** np.floor(tt / dt), color=col, linestyle=ls, linewidth=1.5, label=lab, drawstyle='steps-post')
a1.set_xlabel("tempo fisico senza ingresso [s]"); a1.set_ylabel("potenziale residuo")
a1.xaxis.set_major_formatter(comma(1)); a1.yaxis.set_major_formatter(comma(1))
a1.set_ylim(0, 1.05); a1.set_xlim(0, 3)
a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), fontsize=7, handlelength=1.8, ncol=1)
a1.set_title(r"\textbf{(a)} memoria del neurone", loc='left')

rows = [("finestra fissa", 86.52, CLOCK['E0']['color'], 'o', 'white'),
        ("lunghezza variabile", 94.23, CLOCK['VC']['color'], '^', 'white'),
        ("+ assestamento di 500\\,ms", 94.64, CLOCK['VC']['color'], '^', None),
        ("+ assestamento, $\\tau_d\\approx4{,}91$", 84.03, CLOCK['VC']['color'], '^', None)]
yy = np.arange(len(rows))[::-1]
for yi, (lab, v, col, mk, _) in zip(yy, rows):
    hollow = lab.startswith('+')
    a2.plot([80, v], [yi, yi], color=GRID, linewidth=2.5, solid_capstyle='round', zorder=1)
    a2.plot(v, yi, marker=mk, markersize=7, color='white' if hollow else col, markeredgecolor=col, markeredgewidth=1.3, zorder=3)
    a2.text(v + 0.95, yi, r"$%s$" % num(v, 2), va='center', fontsize=7.5, color=INK)
a2.set_yticks(yy); a2.set_yticklabels([r[0] for r in rows], fontsize=7.5)
a2.tick_params(axis='y', length=0); a2.grid(axis='y', visible=False)
a2.set_xlim(80, 99); a2.xaxis.set_major_formatter(comma(0))
a2.set_xlabel(r"Macro-F1 [\%]")
a2.set_title(r"\textbf{(b)} risultato su DVS128 Gesture", loc='left', x=-0.62)
save(fig, 'cap4_settling_tau')
