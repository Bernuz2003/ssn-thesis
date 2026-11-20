"""Run del capitolo 6 (validazione di sviluppo): capacita', generalizzazione, voto fra varianti.

Metriche da summary.json; media delle ultime 16 epoche (u16) di Macro-F1 di validazione e di accuratezza di
addestramento da history.csv; confronti appaiati con il test esatto di McNemar (riuso di cap5/risultati5.py).
Il voto a maggioranza usa le predizioni di validazione; in caso di parita' vince la classe del primo membro
nell'ordine elencato, senza usare il target (docs/STRUTTURA_NARRATIVA..., par. 11.7).
"""
import csv, json, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cap5'))
import risultati5 as r5  # noqa: E402

RUNS = r5.RUNS
EXTRA = {
    'dvslip': {
        'irm_1m': 'VC_RI_M--interval-antisymmetric-r16-capacity-1m__20261115_231446_051930__seed42',
        'irm_2m': 'VC_RI_M--interval-antisymmetric-r16-capacity-2m__20261115_232156_104565__seed42',
        'kd5': 'VC_RI_M--interval-antisymmetric-r16-kd5__20261112_083122_644838__seed42',
        'kd4': 'VC_RI_M--interval-antisymmetric-r16-kd4__20261119_025813_296654__seed42',
        'prepool': 'VC_RI_M--interval-antisymmetric-r16-local-prepool-r16__20261118_133930_643475__seed42',
        'irm_se': 'VC_RI_M--interval-antisymmetric-r16-se__20261116_020421_893289__seed42',
        'rif_se': 'VC_C0_M--se__20261116_010316_895639__seed42',
    },
    'thu_eact_50_chl': {
        'irm_1m': 'VC_RI_M--interval-antisymmetric-r16-capacity-1m__20261116_001654_325223__seed42',
        'irm_2m': 'VC_RI_M--interval-antisymmetric-r16-capacity-2m__20261116_001738_151053__seed42',
        'kd5': 'VC_RI_M--interval-antisymmetric-r16-kd5__20261112_090209_944284__seed42',
        'prepool': 'VC_RI_M--interval-antisymmetric-r16-local-prepool-r16__20261118_141606_059420__seed42',
        'irm_se': 'VC_RI_M--interval-antisymmetric-r16-se__20261116_092618_894314__seed42',
        'rif_se': 'VC_C0_M--se__20261116_014730_846985__seed42',
    },
}
for ds in EXTRA:
    r5.RUN[ds].update(EXTRA[ds])
RUN = r5.RUN
paired, metrics, path = r5.paired, r5.metrics, r5.path

# Membri del voto (gli stessi insegnanti della distillazione), nell'ordine usato per le parita'
VOTO = {'dvslip': ['antisimmetrico', 'stadio1', 'stadi12', 'punti', 'rif'],
        'thu_eact_50_chl': ['antisimmetrico', 'stadi12', 'instradato', 'punti', 'traccia_sensore']}


def train_u16(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/history.csv')))
    return 100 * np.mean([float(r['train_accuracy']) for r in rows[-16:]])


def predictions(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/validation_predictions.csv')))
    return {r['sample_id']: (int(r['target']), int(r['prediction'])) for r in rows}


def macro_f1(pred, tgt, C):
    f = []
    for c in range(C):
        tp = np.sum((pred == c) & (tgt == c)); fp = np.sum((pred == c) & (tgt != c)); fn = np.sum((pred != c) & (tgt == c))
        p = tp / (tp + fp) if tp + fp else 0.; r = tp / (tp + fn) if tp + fn else 0.
        f.append(2 * p * r / (p + r) if p + r else 0.)
    return 100 * float(np.mean(f))


def voto(ds):
    P = [predictions(ds, k) for k in VOTO[ds]]
    ids = sorted(P[0])
    tgt = np.array([P[0][i][0] for i in ids])
    preds = np.array([[p[i][1] for i in ids] for p in P])           # membri x campioni
    out = []
    for j in range(len(ids)):
        cnt = Counter(preds[:, j]); best = max(cnt.values())
        out.append(next(c for c in preds[:, j] if cnt[c] == best))  # parita': primo membro
    out = np.array(out)
    C = 100 if ds == 'dvslip' else 50
    corr = preds == tgt
    return dict(acc_membri=100 * corr.mean(1), acc_voto=100 * (out == tgt).mean(), f1_voto=macro_f1(out, tgt, C),
                almeno_uno=100 * corr.any(0).mean(), nessuno=100 * (~corr.any(0)).mean(),
                n_corretti=np.bincount(corr.sum(0), minlength=len(P) + 1))


if __name__ == '__main__':
    for ds in RUN:
        print('=====', ds)
        for k in ['rif', 'antisimmetrico', 'irm_1m', 'irm_2m', 'kd5', 'irm_se', 'rif_se']:
            m = metrics(ds, k)
            print(f"  {k:15s} F1 {m['f1']:6.2f} u16 {m['u16']:6.2f} acc {m['acc']:6.2f} train_u16 {train_u16(ds, k):6.2f}"
                  f"  params {json.load(open(path(ds, k) + '/summary.json'))['trainable_parameters']}")
        for a, b in [('antisimmetrico', 'irm_1m'), ('antisimmetrico', 'irm_2m'), ('irm_1m', 'irm_2m'),
                     ('antisimmetrico', 'kd5'), ('antisimmetrico', 'irm_se'), ('rif', 'rif_se'), ('rif', 'antisimmetrico')]:
            x = paired(ds, a, b)
            print(f"  {a:>14s} -> {b:10s} dacc {x['dacc']:+6.2f} {x['fixed']:4d}/{x['broken']:<4d} p={x['p']:.2g}")
        v = voto(ds)
        print('  voto: membri', np.round(v['acc_membri'], 2), 'media %.2f' % v['acc_membri'].mean(),
              'voto acc %.2f F1 %.2f' % (v['acc_voto'], v['f1_voto']),
              'almeno uno %.1f nessuno %.1f' % (v['almeno_uno'], v['nessuno']), 'n corretti', v['n_corretti'])


# ---- errori: gruppi di parole su DVS-Lip, quartili di durata, classi di THU --------------------------------------
def gruppi_lip(key):
    v = json.load(open(f"{path('dvslip', key)}/summary.json"))['validation']['class_group_accuracies']
    acc1, acc2 = v['metrics']['Acc1'], v['metrics']['Acc2']
    pe = v['visually_confusable_pair_errors']
    err1 = round((1 - acc1['accuracy']) * acc1['samples'])
    return dict(acc1=100 * acc1['accuracy'], acc2=100 * acc2['accuracy'], dentro=pe['within_pair_errors'],
                altri1=err1 - pe['within_pair_errors'], err2=round((1 - acc2['accuracy']) * acc2['samples']),
                n1=acc1['samples'], n2=acc2['samples'])


def quartili(ds, keys, ref='rif'):
    """Accuratezza per quartile di durata; i bucket usano i passi validi del modello di riferimento."""
    rows = list(csv.DictReader(open(f'{path(ds, ref)}/validation_predictions.csv')))
    steps = {r['sample_id']: int(r['valid_steps']) for r in rows}
    edges = np.quantile(list(steps.values()), [0.25, 0.5, 0.75])
    bucket = {i: int(np.searchsorted(edges, t, side='left')) for i, t in steps.items()}
    out = {}
    for k in keys:
        P = predictions(ds, k)
        acc = []
        for b in range(4):
            ids = [i for i in P if bucket[i] == b]
            acc.append(100 * np.mean([P[i][0] == P[i][1] for i in ids]))
        out[k] = acc
    n = [sum(1 for i in bucket if bucket[i] == b) for b in range(4)]
    return edges, n, out


def classi_thu(a='rif', b='antisimmetrico'):
    Pa, Pb = predictions('thu_eact_50_chl', a), predictions('thu_eact_50_chl', b)
    ids = sorted(Pa); t = np.array([Pa[i][0] for i in ids])
    pa = np.array([Pa[i][1] for i in ids]); pb = np.array([Pb[i][1] for i in ids])
    def f1(p, c):
        tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.
    cls = [c for c in range(50) if np.sum(t == c)]
    d = np.array([f1(pb, c) - f1(pa, c) for c in cls])
    return len(cls), int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((abs(d) <= 1e-9).sum())
