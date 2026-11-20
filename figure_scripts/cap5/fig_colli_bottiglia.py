"""Due dataset, due colli di bottiglia (sezione 5.5): figura cap5_colli_bottiglia e numeri del testo.

(a) Accuratezza di training (media delle ultime 16 epoche) contro Macro-F1 di validazione per l'IRM antisimmetrico e i
    dieci controlli della sezione 5.4 (controlli5.json, prodotto da fig_controlli.py).
(b) Quota delle clip di validazione riconosciute da 0, 1, ..., 5 delle cinque varianti compatte del voto, le stesse
    architetture degli insegnanti della distillazione (capitolo 6). Su DVS-Lip l'IRM e' la run di riferimento 57,47.
    DVS-Lip: IRM, IRM nel primo patch embedding, IRM in entrambi, ritardi puntuali, modello di riferimento.
    THU-EACT-50-CHL: IRM, IRM in entrambi i patch embedding, IRM con instradamento, ritardi puntuali, tracce guidate dal
    sensore. Voto a maggioranza, parita' al primo membro (come risultati6.voto).
"""
import csv, json, os
from collections import Counter
import numpy as np
from common5 import *
from risultati5 import path

VOTO = {'dvslip': ['antisimmetrico', 'stadio1', 'stadi12', 'punti', 'rif'],
        'thu_eact_50_chl': ['antisimmetrico', 'stadi12', 'instradato', 'punti', 'traccia_sensore']}
COL = {'dvslip': '#2a78d6', 'thu_eact_50_chl': '#eb6834'}
NAME = {'dvslip': 'DVS-Lip', 'thu_eact_50_chl': 'THU-EACT-50-CHL'}


def preds(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/validation_predictions.csv')))
    return {r['sample_id']: (int(r['target']), int(r['prediction'])) for r in rows}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def vote(ds):
    P = [preds(ds, k) for k in VOTO[ds]]
    ids = sorted(P[0]); t = np.array([P[0][i][0] for i in ids])
    M = np.array([[p[i][1] for i in ids] for p in P])
    v = []
    for j in range(len(ids)):
        c = Counter(M[:, j]); b = max(c.values()); v.append(next(x for x in M[:, j] if c[x] == b))
    corr = M == t
    hist = np.bincount(corr.sum(0), minlength=6) / len(ids) * 100
    return dict(members=100 * corr.mean(1), vote=100 * np.mean(np.array(v) == t), hist=hist)


S = json.load(open(os.path.join(HERE, 'controlli5.json')))
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), gridspec_kw=dict(wspace=0.3, width_ratios=[1.1, 1]))

ax = axes[0]
for ds in ('dvslip', 'thu_eact_50_chl'):
    keys = list(S[ds])
    tr = np.array([S[ds][k]['train'] for k in keys]); f1 = np.array([S[ds][k]['f1'] for k in keys])
    rho = spearman(tr, f1)
    print(f"{ds}: train {tr.min():.1f}-{tr.max():.1f}, val F1 {f1.min():.1f}-{f1.max():.1f}, Spearman {rho:+.2f}")
    other = [k != 'antisimmetrico' for k in keys]
    ax.scatter(tr[other], f1[other], s=22, color=COL[ds], alpha=0.85, edgecolor='white', lw=0.6, zorder=3,
               label=NAME[ds])
    i = keys.index('antisimmetrico')
    ax.scatter(tr[i], f1[i], s=60, marker='*', color=COL[ds], edgecolor=INK, lw=0.5, zorder=4)
    ax.annotate('IRM', (tr[i], f1[i]), xytext=(-4, 5), textcoords='offset points', ha='right', fontsize=6.8,
                color=INK)
ax.set_xlabel(r"accuratezza di training [\%]", fontsize=7.5)
ax.set_ylabel(r"Macro-F1 di validazione [\%]", fontsize=7.5)
ax.set_xlim(62, 101); ax.set_ylim(49, 64.5)
ax.xaxis.set_major_formatter(comma(0)); ax.yaxis.set_major_formatter(comma(0))
ax.legend(loc='lower right', fontsize=7, frameon=False, handletextpad=0.2)
ax.set_title(r"\textbf{(a)} adattamento e validazione", loc='left', fontsize=8.2)

ax = axes[1]
x = np.arange(6)
for j, ds in enumerate(('dvslip', 'thu_eact_50_chl')):
    r = vote(ds)
    print(f"{ds}: membri {np.round(r['members'], 2)}, media {r['members'].mean():.2f}, voto {r['vote']:.2f}, "
          f"istogramma {np.round(r['hist'], 1)}")
    ax.bar(x + (j - 0.5) * 0.38, r['hist'], width=0.36, color=COL[ds], label=NAME[ds], zorder=2)
ax.set_xticks(x)
ax.set_xlabel("varianti che riconoscono la clip", fontsize=7.5)
ax.set_ylabel(r"clip di validazione [\%]", fontsize=7.5)
ax.yaxis.set_major_formatter(comma(0))
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.legend(loc='upper left', fontsize=7, frameon=False, handletextpad=0.3)
ax.set_title(r"\textbf{(b)} accordo fra cinque varianti", loc='left', fontsize=8.2)
save(fig, 'cap5_colli_bottiglia')
