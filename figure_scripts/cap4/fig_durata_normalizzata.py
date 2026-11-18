"""Controllo a durata normalizzata (sezione 4.3): accuratezza per quartili di durata su DVS-Lip e THU-EACT-50-CHL.

Finestra fissa e lunghezza variabile: valutazione appaiata dei checkpoint (reports/c0_e0_vc, come nella figura dei
quartili della sezione 4.2). Durata normalizzata: predizioni salvate alla fine dell'addestramento
(runs/<dataset>/FBC_C0_M--vc-horizon__*/validation_predictions.csv), stesso frazionamento in microbatch e stessi
ritardi riscalati della lunghezza variabile. Quartili calcolati sulle durate della validazione; sotto ogni quartile
la durata mediana e la durata del bin della durata normalizzata (durata / 40).
"""
import csv, glob
import numpy as np
from matplotlib.lines import Line2D
from style import *
from load import load

RUNS = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/runs'
SETS = [('dvslip', 'DVS-Lip'), ('thu_eact_50_chl', 'THU-EACT-50-CHL')]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45), gridspec_kw=dict(wspace=0.28))
for ax, (d, name) in zip(axes, SETS):
    E = load(d, 'E0'); V = load(d, 'VC')
    path = glob.glob(f'{RUNS}/{d}/FBC_C0_M--vc-horizon__*/validation_predictions.csv')
    assert len(path) == 1, path
    fbc = {r['sample_id']: int(r['target']) == int(r['prediction']) for r in csv.DictReader(open(path[0]))}
    ids = E['ids']
    assert ids == V['ids'] and set(ids) == set(fbc)
    correct = {'E0': E['correct'], 'VC': V['correct'], 'FBC': np.array([fbc[i] for i in ids])}
    dur = E['duration']
    q = np.quantile(dur, [.25, .5, .75]); b = np.digitize(dur, q)
    meds = [np.median(dur[b == i]) / 1e6 for i in range(4)]
    print(name, {k: round(100 * v.mean(), 2) for k, v in correct.items()})
    for key, off in (('E0', -0.13), ('FBC', 0.0), ('VC', 0.13)):
        c = CLOCK[key]; acc = []
        for i in range(4):
            m = b == i; k, n = correct[key][m].sum(), m.sum(); lo_, hi_ = wilson(k, n)
            ax.plot([i + off, i + off], [lo_ * 100, hi_ * 100], color=c['color'], linewidth=1.0, alpha=0.45)
            acc.append(k / n * 100)
        ax.plot(np.arange(4) + off, acc, color=c['color'], marker=c['marker'], markersize=4.6, linewidth=1.4,
                markeredgecolor='white', markeredgewidth=0.6)
        print(f"  {key:4s}", [round(a, 1) for a in acc])
    print("  durate mediane [s]", [round(m, 2) for m in meds], " bin [ms]", [round(1e3 * m / 40, 1) for m in meds])
    ax.set_xticks(range(4))
    ax.set_xticklabels([r"$%s$\,s" "\n" r"$%s$\,ms" % (num(m, 2 if m < 2 else 1), num(1e3 * m / 40, 0)) for m in meds],
                       fontsize=7)
    ax.set_xlim(-0.45, 3.45)
    ax.yaxis.set_major_formatter(comma(0))
    ax.set_title(name, fontsize=8.5)
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel(r"accuratezza [\%]")
fig.text(0.5, -0.07, r"durata mediana del quartile e durata del bin con la durata normalizzata", ha='center', fontsize=8.5)
handles = [Line2D([], [], marker=CLOCK[k]['marker'], color=CLOCK[k]['color'], markersize=5, label=CLOCK[k]['label'])
           for k in ('E0', 'FBC', 'VC')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.5)
save(fig, 'cap4_durata_normalizzata')
