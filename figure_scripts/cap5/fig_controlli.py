"""Controlli dell'IRM antisimmetrico (sezione 5.4): statistiche e forest plot cap5_controlli.

Per ogni controllo e dataset, sulla validazione di sviluppo (predizioni salvate durante l'addestramento):
  - Macro-F1 del controllo e differenza rispetto all'IRM antisimmetrico, con intervallo bootstrap appaiato al 95%
    (2000 ricampionamenti delle clip, percentili, seme 0); la Macro-F1 media tutte le classi della testa;
  - p del test esatto di McNemar sull'accuratezza;
  - accuratezza di training, media delle ultime 16 epoche (usata nella sezione 5.5).
Scrive controlli5.json accanto allo script. Le differenze coincidono con il report
artifacts/reports/b_minus_final_inference/validation_controls_vs_bminus.txt, entro la variabilita' del bootstrap.
"""
import csv, json, os
import numpy as np
from common5 import *
from risultati5 import RUN, path, mcnemar

CLASSES = {'dvslip': 100, 'thu_eact_50_chl': 50}
GROUPS = [
    ('Capacità', [('rif', 'modello di riferimento'), ('vuoto', 'sito vuoto'), ('mlp', 'MLP di pari parametri')]),
    ('Confronto', [('lineare', 'bande senza confronto'), ('simmetrico', 'confronto simmetrico')]),
    ('Organizzazione del passato', [('punti', 'ritardi puntuali'), ('traccia_statica', 'tracce a decadimento appreso'),
                 ('traccia_sensore', 'tracce guidate dal sensore'), ('orologio', 'orologio appreso'),
                 ('remota', 'passato remoto aggiunto')]),
]
OUTJ = os.path.join(HERE, 'controlli5.json')


def preds(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/validation_predictions.csv')))
    return {r['sample_id']: (int(r['target']), int(r['prediction'])) for r in rows}


def f1_matrix(t, p, C):
    """Macro-F1 per riga: t e p hanno forma (B, N)."""
    out = np.zeros(t.shape[0])
    for c in range(C):
        tp = np.sum((p == c) & (t == c), 1); fp = np.sum((p == c) & (t != c), 1); fn = np.sum((p != c) & (t == c), 1)
        den = 2 * tp + fp + fn
        out += np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.)
    return 100 * out / C


def train_u16(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/history.csv')))
    col = [k for k in rows[0] if k.startswith('train') and 'acc' in k][0]
    return 100 * np.mean([float(r[col]) for r in rows[-16:]])


def stats():
    rng = np.random.default_rng(0)
    out = {}
    for ds in ('dvslip', 'thu_eact_50_chl'):
        C = CLASSES[ds]
        base = preds(ds, 'antisimmetrico'); ids = sorted(base)
        t = np.array([base[i][0] for i in ids]); pb = np.array([base[i][1] for i in ids])
        idx = rng.integers(0, len(ids), size=(2000, len(ids)))
        f1b = f1_matrix(t[None], pb[None], C)[0]
        out[ds] = {'antisimmetrico': dict(f1=f1b, train=train_u16(ds, 'antisimmetrico'))}
        for _, rows in GROUPS:
            for key, _ in rows:
                q = preds(ds, key); assert set(q) == set(ids)
                pq = np.array([q[i][1] for i in ids])
                f1q = f1_matrix(t[None], pq[None], C)[0]
                boot = f1_matrix(t[idx], pq[idx], C) - f1_matrix(t[idx], pb[idx], C)
                fixed = int(np.sum((pq == t) & (pb != t))); broken = int(np.sum((pq != t) & (pb == t)))
                out[ds][key] = dict(f1=f1q, d=f1q - f1b, lo=float(np.percentile(boot, 2.5)),
                                    hi=float(np.percentile(boot, 97.5)), p=mcnemar(fixed, broken),
                                    train=train_u16(ds, key))
    json.dump(out, open(OUTJ, 'w'), indent=1)
    return out


if __name__ == '__main__':
    S = stats()
    for ds in S:
        print('=====', ds, 'IRM F1 %.2f train %.1f' % (S[ds]['antisimmetrico']['f1'], S[ds]['antisimmetrico']['train']))
        for _, rows in GROUPS:
            for key, lab in rows:
                r = S[ds][key]
                print(f"  {lab:32s} F1 {r['f1']:6.2f}  d {r['d']:+6.2f} [{r['lo']:+5.1f};{r['hi']:+5.1f}]  p {r['p']:.3g}  train {r['train']:5.1f}")

    # ---- forest plot -------------------------------------------------------------------------------
    labels, keys, gpos = [], [], []
    y = 0
    for g, rows in GROUPS:
        gpos.append((g, y))
        for key, lab in rows:
            keys.append((key, y)); labels.append(lab); y += 1
        y += 0.6
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (ds, name) in zip(axes, [('dvslip', 'DVS-Lip'), ('thu_eact_50_chl', 'THU-EACT-50-CHL')]):
        for (key, yy) in keys:
            r = S[ds][key]
            sig = r['p'] < 0.05
            ax.plot([r['lo'], r['hi']], [yy, yy], color=INK2, lw=1.0, zorder=2)
            ax.plot(r['d'], yy, 'o', ms=5, mfc=('#2a78d6' if sig else 'white'), mec='#2a78d6', mew=1.2, zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_title(name, fontsize=8.5)
        for g, yy in gpos[1:]:
            ax.axhline(yy - 1.3, color=GRID if 'GRID' in globals() else '#dddddd', lw=0.6, zorder=0)
        ax.xaxis.set_major_formatter(comma(0))
        ax.grid(axis='y', visible=False)
        if ds == 'dvslip':  # scala piu' stretta, per leggere le differenze piccole
            ax.set_xlim(-5.5, 5.5); ax.set_xticks([-4, -2, 0, 2, 4])
        else:
            ax.set_xlim(-17, 6)
    axes[0].set_yticks([yy for _, yy in keys])
    axes[0].set_yticklabels(labels, fontsize=7.3)
    axes[0].invert_yaxis()
    for g, yy in gpos:
        axes[0].text(-0.02, yy - 0.75, r'\textbf{%s}' % g, transform=axes[0].get_yaxis_transform(), ha='right',
                     va='center', fontsize=7.3, color=INK)
    axes[0].tick_params(axis='y', length=0)
    fig.supxlabel("differenza di Macro-F1 rispetto all'IRM antisimmetrico [punti]", fontsize=7.5, y=0.0)
    save(fig, 'cap5_controlli')
