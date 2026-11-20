"""Compressione dopo l'addestramento a ogni capacita' (sezione 6.3): numeri della tabella e del testo.

Fonte: artifacts/reports/b_minus_final_inference/<worker>.json e .npz (sola inferenza). Il rango e' scelto sulla
validazione di sviluppo con la regola del manifest: il piu' piccolo q fra 8, 16, 24, 32, 48, 64, 96, 128, 192 (minori
della larghezza) la cui Macro-F1 di validazione e' almeno quella del denso meno 0,5 punti, valutata nella stessa
esecuzione. Costi per frame (32 clip di validazione) con la funzione cost() di notebooks/b_minus_thesis.py: MAC e stato
del sito = somma sui livelli del secondo patch embedding; lo stato e' convertito in valori (32 bit ciascuno).
"""
import json
from math import comb
import numpy as np

REP = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/b_minus_final_inference'
PREFIX = ("patch_embed2.down.temporal_channel_mixer", "patch_embed2.down.recent", "patch_embed2.down.remote")
MODELS = {
    'thu_eact_50_chl': [('IRM', '0,5 M', ['thu_b']), ('IRM', '1 M', ['thu_b_1m']),
                        ('IRM', '2 M', ['thu_b_2m', 'thu_b_2m_s43', 'thu_b_2m_s44'])],
    'dvslip': [('IRM', '0,5 M', ['lip_b']), ('IRM', '1 M', ['lip_b_1m']),
               ('IRM', '2 M', ['lip_b_2m', 'lip_b_2m_s43', 'lip_b_2m_s44']),
               ('IRM distillato', '0,5 M', ['lip_kd4_05']),
               ('IRM distillato', '1 M', ['lip_kd4_1m']),
               ('IRM distillato', '2 M', ['lip_kd4_2m', 'lip_kd4_2m_s43', 'lip_kd4_2m_s44']),
               ('IRM con relazioni locali', '1 M', ['lip_prepool_1m']),
               ('IRM con relazioni locali', '2 M', ['lip_prepool_2m']),
               ('IRM con relazioni locali, distillato', '0,5 M', ['lip_prepool_kd4_05']),
               ('IRM con relazioni locali, distillato', '1 M', ['lip_prepool_1m_kd4']),
               ('IRM con relazioni locali, distillato', '2 M', ['lip_prepool_2m_kd4'])],
}
MIXER = {'thu_eact_50_chl': {'0,5 M': 'thu_c0', '1 M': 'thu_c0_1m', '2 M': 'thu_c0_2m'},
         'dvslip': {'0,5 M': 'lip_c0', '1 M': 'lip_c0_1m', '2 M': 'lip_c0_2m'}}


def load(w):
    return json.load(open(f'{REP}/{w}.json'))


def cost(profile):
    layers, state = profile['layers_per_frame'], profile['state_bits']
    mod = {k: v for k, v in layers.items() if k.startswith(PREFIX)}
    return dict(params=profile['params_total'], mac=profile['totals_per_frame'].get('multivalued_mac_potential', 0.),
                state=sum(state.values()) / 32, module_params=profile.get('module_params', 0),
                module_mac=sum(v.get('temporal_channel_mixer_mac', v.get('multivalued_mac_potential', 0.))
                               for v in mod.values()),
                module_state=sum(v for k, v in state.items() if k.startswith(PREFIX)) / 32)


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def worker(w):
    r = load(w)
    q = r.get('chosen_rank')
    d = cost(r['arms']['dense']['profile'])
    out = dict(q=q, dense=d, val_dense=100 * r['arms']['dense']['validation']['macro_f1'],
               test_dense=100 * r['arms']['dense']['test']['accuracy'])
    out['ranks'] = sorted(int(a[1:]) for a in r['arms'] if a.startswith('q'))
    out['val_curve'] = {k: 100 * r['arms'][f'q{k}']['validation']['macro_f1'] - out['val_dense'] for k in out['ranks']}
    out['energy'] = {k: 100 * r['arms'][f'q{k}']['factorization']['retained_weight_energy'] for k in out['ranks']}
    if q is not None:
        out['comp'] = cost(r['arms'][f'q{q}']['profile'])
        out['test_comp'] = 100 * r['arms'][f'q{q}']['test']['accuracy']
        z = np.load(f'{REP}/{w}.npz'); t = z['test_targets']
        a = z['test_logits_dense'].argmax(1) == t; b = z[f'test_logits_q{q}'].argmax(1) == t
        out['p'] = mcnemar(int(np.sum(b & ~a)), int(np.sum(a & ~b)))
    return out


def saving(d, c, k):
    return 100 * (1 - c[k] / d[k])


if __name__ == '__main__':
    for ds, rows in MODELS.items():
        print('=====', ds)
        for lab, size, ws in rows:
            for w in ws:
                o = worker(w)
                if o['q'] is None:
                    best = max((k for k in o['ranks']), key=lambda k: o['val_curve'][k])
                    print(f"  {lab:15s} {size:14s} {w:16s} nessun rango; migliore q{best} {o['val_curve'][best]:+.2f}"
                          f"  test {o['test_dense']:.2f}")
                    continue
                d, c = o['dense'], o['comp']
                print(f"  {lab:15s} {size:14s} {w:16s} q{o['q']:<4d} energia {o['energy'][o['q']]:.1f}%"
                      f"  modulo MAC {d['module_mac']/1e6:6.2f}->{c['module_mac']/1e6:6.2f} (-{saving(d,c,'module_mac'):.1f}%)"
                      f"  stato {d['module_state']/1e3:6.1f}->{c['module_state']/1e3:6.1f} k (-{saving(d,c,'module_state'):.1f}%)"
                      f"  par {d['params']}->{c['params']} (-{saving(d,c,'params'):.1f}%)"
                      f"  rete MAC -{saving(d,c,'mac'):.1f}% stato -{saving(d,c,'state'):.1f}%"
                      f"  test {o['test_dense']:.2f}->{o['test_comp']:.2f} p={o['p']:.2f}")
        for size, w in MIXER[ds].items():
            m = cost(load(w)['arms']['dense']['profile'])
            print(f"  mixer a ritardi {size:5s} {w:12s} params {m['params']} modulo par {m['module_params']}"
                  f" MAC {m['module_mac']/1e6:.2f} stato {m['module_state']/1e3:.1f} k  rete MAC {m['mac']/1e6:.1f}"
                  f" stato {m['state']/1e3:.0f} k  test {100*load(w)['arms']['dense']['test']['accuracy']:.2f}")
