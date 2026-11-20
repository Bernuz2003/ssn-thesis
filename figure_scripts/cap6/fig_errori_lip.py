"""Errori residui su DVS-Lip (sezione 6.6): figure cap6_confusione_lip e cap6_durata.

cap6_confusione_lip
(a) Matrice degli errori sul test ufficiale dell'IRM con due milioni di parametri e distillazione (conteggi sommati sui
    tre addestramenti), normalizzata per riga e con la diagonale esclusa. Le 100 parole sono ordinate mettendo accanto le
    due parole di ciascuna delle 25 coppie visivamente simili (gruppo di Acc1), seguite dalle altre 50 (gruppo di Acc2),
    secondo configs/dvslip_class_groups.json.
(b) Errori sul test per gruppo, lungo il percorso dei modelli: scambi dentro la coppia, altri errori sulle parole delle
    coppie, errori sulle altre parole (media sugli addestramenti ripetuti).
cap6_durata
    Accuratezza per quartile del numero di passi validi, in validazione (passi dal file delle predizioni del modello di
    riferimento) e sul test (passi salvati da artifacts/reports/c0_ri_diagnostics/workers/dvslip_c0/original).
"""
import csv, json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from common6 import *
from finali6 import test
from risultati6 import path

G = json.load(open('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/configs/dvslip_class_groups.json'))
CLS = G['classes']; IDX = {c: i for i, c in enumerate(CLS)}
PAIRS = G['visually_confusable_pairs']
ORDER = [IDX[w] for p in PAIRS for w in p] + sorted(IDX[w] for w in G['common_words'])
PARTNER = -np.ones(100, int)
for a, b in PAIRS:
    PARTNER[IDX[a]] = IDX[b]; PARTNER[IDX[b]] = IDX[a]
CONF = PARTNER >= 0
PATH = [('riferi-\nmento', ['lip_c0']), ('IRM', ['lip_b']), ('IRM\ndistillato', ['lip_kd5']),
        ('IRM\n2 M', ['lip_b_2m', 'lip_b_2m_s43', 'lip_b_2m_s44']),
        ('IRM 2 M\ndistillato', ['lip_kd4_2m', 'lip_kd4_2m_s43', 'lip_kd4_2m_s44'])]
BLUE = DSCOL['dvslip']

# ---- confusione ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.15), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.05]))
ax = axes[0]
C = np.zeros((100, 100))
for w in PATH[-1][1]:
    t = test(w); np.add.at(C, (t['target'], t['pred']), 1)
R = C / C.sum(1, keepdims=True)
M = R[np.ix_(ORDER, ORDER)].copy()
np.fill_diagonal(M, np.nan)
cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#9dbfe8', '#2a78d6', '#0d3d78'])
cmap.set_bad('#d9d9d9')
im = ax.imshow(100 * M, cmap=cmap, vmin=0, vmax=40, interpolation='nearest')
for v in (49.5,):
    ax.axhline(v, color=INK, lw=0.6); ax.axvline(v, color=INK, lw=0.6)
ax.set_xticks([24.5, 74.5]); ax.set_xticklabels(['25 coppie simili', 'altre 50 parole'], fontsize=6.8)
ax.set_yticks([24.5, 74.5]); ax.set_yticklabels(['25 coppie\nsimili', 'altre 50\nparole'], fontsize=6.8)
ax.tick_params(length=0)
ax.set_xlabel("parola predetta", fontsize=7.5); ax.set_ylabel("parola pronunciata", fontsize=7.5)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6.5)
cb.ax.yaxis.set_major_formatter(comma(0))
ax.set_title(r"\textbf{(a)} errori dell'IRM 2 M distillato", loc='left', fontsize=8.2)

ax = axes[1]
vals = []
for lab, ws in PATH:
    r = []
    for w in ws:
        t = test(w); y, p = t['target'], t['pred']; err = p != y; g1 = CONF[y]
        swap = err & g1 & (p == PARTNER[y])
        r.append((swap.sum(), (err & g1 & ~swap).sum(), (err & ~g1).sum()))
    vals.append(np.mean(r, 0))
vals = np.array(vals)
print('errori (scambi, altri gruppo 1, gruppo 2):'); print(np.round(vals, 1))
x = np.arange(len(PATH))
cols = ['#0d3d78', '#7fb0ea', '#d0d0d0']
labs = ['scambi dentro la coppia', 'altri errori, coppie simili', 'errori, altre parole']
bottom = np.zeros(len(PATH))
for k in range(3):
    ax.bar(x, vals[:, k], bottom=bottom, width=0.62, color=cols[k], label=labs[k], zorder=2,
           edgecolor='white', linewidth=0.6)
    for xi, b0, v in zip(x, bottom, vals[:, k]):
        ax.text(xi, b0 + v / 2, f"{v:.0f}", ha='center', va='center', fontsize=6.3,
                color='white' if k == 0 else INK)
    bottom += vals[:, k]
ax.set_xticks(x); ax.set_xticklabels([l for l, _ in PATH], fontsize=6.6)
ax.set_ylabel("clip sbagliate sul test", fontsize=7.5)
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.set_ylim(0, 3200)
ax.legend(loc='upper right', fontsize=6.3, frameon=False)
ax.set_title(r"\textbf{(b)} errori lungo il percorso", loc='left', fontsize=8.2)
save(fig, 'cap6_confusione_lip')

# ---- durata ----------------------------------------------------------------------------------------------------
z = np.load('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/c0_ri_diagnostics/'
            'workers/dvslip_c0/original/predictions.npz')
steps_test = dict(zip(z['sample_ids'].tolist(), z['steps'].tolist()))
rows = list(csv.DictReader(open(f"{path('dvslip', 'rif')}/validation_predictions.csv")))
steps_val = {r['sample_id']: int(r['valid_steps']) for r in rows}
COLS = ['#a8a8a8', '#7fb0ea', '#7fb0ea', '#0d3d78', '#0d3d78']  # grigio riferimento, chiaro 0,5 M, scuro 2 M
STY = ['-', '-', '--', '-', '--']  # tratteggio: distillazione
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, part, steps, title in [(axes[0], 'val', steps_val, 'validazione di sviluppo'),
                               (axes[1], 'test', steps_test, 'test ufficiale')]:
    ids = test('lip_c0', part=part)['ids'].tolist()
    st = np.array([steps[i] for i in ids]); e = np.quantile(st, [.25, .5, .75])
    b = np.searchsorted(e, st, side='left')
    print(part, 'soglie in passi', e, 'numerosita', np.bincount(b))
    for (lab, ws), col, ls in zip(PATH, COLS, STY):
        q = []
        for w in ws:
            t = test(w, part=part); assert t['ids'].tolist() == ids
            c = t['pred'] == t['target']; q.append([100 * c[b == j].mean() for j in range(4)])
        q = np.mean(q, 0)
        print(f"  {lab.replace(chr(10), ' '):22s}", np.round(q, 1))
        ax.plot(np.arange(4), q, ls=ls, marker='o', ms=4, lw=1.4, color=col,
                label=lab.replace('-\n', '').replace('\n', ' '))
    ax.set_xticks(range(4)); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=7.5)
    ax.set_xlabel("quartile di durata, dal più breve", fontsize=7.5)
    ax.set_title(title, fontsize=8.5)
    ax.yaxis.set_major_formatter(comma(0))
axes[0].set_ylabel(r"accuratezza [\%]", fontsize=7.5)
axes[0].legend(loc='upper left', fontsize=6.4, frameon=False)
save(fig, 'cap6_durata')
