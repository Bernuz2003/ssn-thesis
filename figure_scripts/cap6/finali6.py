"""Metriche sul test ufficiale dei modelli del capitolo 6, dai logit di
artifacts/reports/b_minus_final_inference/<worker>.npz (valutazione a posteriori, sola inferenza).

test(worker, arm='dense') -> dict(acc, f1, pred, target). La Macro-F1 media tutte le classi della testa.
Eseguito da solo stampa la tabella di controllo contro i valori del documento di struttura.
"""
import numpy as np

REP = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/b_minus_final_inference'
CLASSES = {'lip': 100, 'thu': 50}
PARAMS = {
    'lip_c0': 501028, 'lip_b': 513324, 'lip_kd5': 513324, 'lip_kd4_05': 513324,
    'lip_c0_1m': 1114468, 'lip_b_1m': 1132908, 'lip_kd4_1m': 1132908,
    'lip_c0_2m': 1969892, 'lip_b_2m': 1994476, 'lip_kd4_2m': 1994476, 'lip_c0_2m_kd4': 1969892,
    'thu_c0': 494578, 'thu_b': 506874, 'thu_c0_1m': 1104818, 'thu_b_1m': 1123258,
    'thu_c0_2m': 1957042, 'thu_b_2m': 1981626,
}
REPEATS = {'lip_b_2m': ['lip_b_2m', 'lip_b_2m_s43', 'lip_b_2m_s44'],
           'lip_kd4_2m': ['lip_kd4_2m', 'lip_kd4_2m_s43', 'lip_kd4_2m_s44'],
           'thu_b_2m': ['thu_b_2m', 'thu_b_2m_s43', 'thu_b_2m_s44'],
           'thu_c0_2m': ['thu_c0_2m', 'thu_c0_2m_s43', 'thu_c0_2m_s44']}


def macro_f1(t, p, C):
    f = []
    for c in range(C):
        tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.)
    return 100 * float(np.mean(f))


def test(worker, arm='dense', part='test'):
    z = np.load(f'{REP}/{worker}.npz')
    t = z[f'{part}_targets']; p = z[f'{part}_logits_{arm}'].argmax(1)
    C = CLASSES[worker.split('_')[0]]
    return dict(acc=100 * float(np.mean(p == t)), f1=macro_f1(t, p, C), pred=p, target=t,
                ids=z[f'{part}_sample_ids'])


def group(worker):
    """Media e deviazione standard sugli addestramenti ripetuti (o il singolo valore)."""
    ws = REPEATS.get(worker, [worker])
    a = np.array([test(w)['acc'] for w in ws]); f = np.array([test(w)['f1'] for w in ws])
    return dict(acc=a.mean(), acc_sd=a.std(ddof=1) if len(a) > 1 else 0., f1=f.mean(),
                f1_sd=f.std(ddof=1) if len(f) > 1 else 0., n=len(ws))


if __name__ == '__main__':
    for w in PARAMS:
        g = group(w)
        print(f"{w:12s} n={g['n']} acc {g['acc']:6.2f} ± {g['acc_sd']:.2f}  F1 {g['f1']:6.2f} ± {g['f1_sd']:.2f}")
