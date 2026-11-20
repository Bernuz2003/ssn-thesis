"""Che cosa usa la soluzione appresa (sezione 5.3): figura cap5_uso_memoria, quattro pannelli.

(a) Variazione di accuratezza sulla validazione di sviluppo quando si azzerano W_k e U_k di una sola banda nel
    secondo patch embedding (intervento sul checkpoint, sola inferenza).
    DVS-Lip: artifacts/reports/lip_ref_bands/dvslip/ri/<condizione>/metrics.json, checkpoint di riferimento
    VC_RI_M--interval-antisymmetric-r16__20261112_082513_589585__seed42 (valutazione del 19/11/2026).
    THU-EACT-50-CHL: artifacts/reports/rt_fbc_comparison/ablations.json, chiave 'thu_eact_50_chl|ri'.
(b) Accuratezza sul test ufficiale del modello di riferimento e dell'IRM antisimmetrico con i passi nell'ordine
    originale, invertito e rimescolato (media di tre permutazioni).
    Modello di riferimento (entrambi i dataset) e IRM su THU: artifacts/reports/c0_ri_diagnostics/csv/E2.csv.
    IRM su DVS-Lip: artifacts/reports/lip_ref_diagnostics/workers/dvslip_ri_ref/<condizione>/predictions.npz.
    Per l'ordine originale dell'IRM su DVS-Lip si usa il valore della tabella del test ufficiale
    (b_minus_final_inference, 50,81%), la stessa grandezza della tabella 5.x; la passata 'original' della diagnostica
    da' 50,85% (due clip su 4975 diverse per effetti numerici dell'inferenza).
Se una condizione Lip manca ancora, la barra viene omessa e il pannello lo segnala.
"""
import csv, json, os
import numpy as np
from common5 import *

REP = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports'
SETS = [('dvslip', 'DVS-Lip', ['1--2', '3--4', '5--7', '8--15']),
        ('thu_eact_50_chl', 'THU-EACT-50-CHL', ['1--3', '4--6', '7--11', '12--23'])]
ANTI = "#2a78d6"


def band_drops(ds):
    if ds == 'dvslip':
        get = lambda c: json.load(open(f'{REP}/lip_ref_bands/dvslip/ri/{c}/metrics.json'))['accuracy']
    else:
        abl = json.load(open(f'{REP}/rt_fbc_comparison/ablations.json'))['thu_eact_50_chl|ri']['conditions']
        get = lambda c: abl[c]['accuracy']
    clean = get('clean')
    return [100 * (get(f'trace_{k}_off') - clean) for k in range(4)]


def order_acc():
    rows = list(csv.DictReader(open(f'{REP}/c0_ri_diagnostics/csv/E2.csv')))
    acc = {(r['dataset'], r['modello'], r['condizione']): float(r['accuracy']) for r in rows}
    out = {}
    for ds, _, _ in SETS:
        out[(ds, 'C0')] = [acc[(ds, 'C0', c)] for c in ('original', 'reverse', 'shuffle_mean')]
        if ds == 'thu_eact_50_chl':
            out[(ds, 'B-')] = [acc[(ds, 'B-', c)] for c in ('original', 'reverse', 'shuffle_mean')]
    root = f'{REP}/lip_ref_diagnostics/workers/dvslip_ri_ref'

    def lip(c):
        p = f'{root}/{c}/predictions.npz'
        if not os.path.exists(f'{root}/{c}/complete.json'):
            return None
        z = np.load(p)
        return 100 * float(np.mean(z['predictions'] == z['targets']))
    final = np.load(f'{REP}/b_minus_final_inference/lip_b.npz')
    original = 100 * float(np.mean(final['test_logits_dense'].argmax(1) == final['test_targets']))
    shuffles = [lip(f'shuffle_{s}') for s in range(3)]
    out[('dvslip', 'B-')] = [original, lip('reverse'),
                             None if None in shuffles else float(np.mean(shuffles))]
    return out


fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.7), gridspec_kw=dict(wspace=0.1, hspace=0.62))

# ---- (a) bande ------------------------------------------------------------------
for ax, (ds, name, labels) in zip(axes[0], SETS):
    d = band_drops(ds)
    x = np.arange(4)
    ax.bar(x, d, width=0.62, color=BAND, edgecolor=[INK2 if k == 3 else BAND[k] for k in range(4)],
           linewidth=0.6, zorder=2)
    for xi, v in zip(x, d):
        ax.text(xi, v - 0.8, r"$%s$" % num(v, 1), ha='center', va='top', fontsize=7.2, color=INK)
    ax.axhline(0, color=INK2, lw=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"banda {k + 1}\n{lab}" for k, lab in enumerate(labels)], fontsize=7.2)
    ax.set_xlabel("intervallo di ritardi della banda [passi]", fontsize=7.5)
    ax.set_ylim(-29, 1.5)
    ax.yaxis.set_major_formatter(comma(0))
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(name, fontsize=8.5)
axes[0][1].set_yticklabels([])
axes[0][0].set_ylabel("variazione di accuratezza\n[punti]", fontsize=7.5)

# ---- (b) ordine dei passi ---------------------------------------------------------
acc = order_acc()
COND = ['originale', 'invertito', 'rimescolato']
MOD = [('C0', 'modello di riferimento', NEUTRAL), ('B-', 'IRM antisimmetrico', ANTI)]
for ax, (ds, name, _) in zip(axes[1], SETS):
    x = np.arange(len(COND))
    for j, (m, lab, col) in enumerate(MOD):
        xs = x + (j - 0.5) * 0.34
        for xi, vi in zip(xs, acc[(ds, m)]):
            if vi is None:
                ax.text(xi, 2, "in corso", rotation=90, ha='center', va='bottom', fontsize=6.5, color=INK2)
                continue
            ax.bar(xi, vi, width=0.32, color=col, label=lab, zorder=2)
            ax.text(xi, vi + 1.2, r"$%s$" % num(vi, 1), ha='center', va='bottom', fontsize=6.8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(COND, fontsize=7.5)
    ax.set_xlabel("ordine dei passi", fontsize=7.5)
    ax.set_ylim(0, 78)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0)
    ax.set_title(name, fontsize=8.5)
axes[1][1].set_yticklabels([])
axes[1][0].set_ylabel(r"accuratezza sul test [\%]", fontsize=7.5)
axes[1][0].yaxis.set_major_formatter(comma(0))
h, l = axes[1][0].get_legend_handles_labels()
keep = dict(zip(l, h))
fig.legend(list(keep.values()), list(keep.keys()), loc='lower center', bbox_to_anchor=(0.53, -0.04), ncol=2,
           fontsize=7.2)
for ax, tag in [(axes[0][0], '(a)'), (axes[1][0], '(b)')]:
    ax.text(-0.2, 1.12, tag, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')
save(fig, 'cap5_uso_memoria')
