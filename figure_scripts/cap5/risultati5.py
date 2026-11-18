"""Run del capitolo 5 (validazione di sviluppo) e confronti appaiati.

Su DVS-Lip l'IRM antisimmetrico di riferimento e' la run 20261112_082513 (Macro-F1 57,47), addestrata nelle stesse
condizioni del modello di riferimento e dei controlli (docs/STRUTTURA_NARRATIVA_E_RISULTATI_CAPITOLI_5_6_7.md, par. 2.1).
Media delle ultime 16 epoche (u16) da history.csv; confronti appaiati con il test esatto di McNemar sulle
predizioni di validazione (validation_predictions.csv).
"""
import csv, json, math, os
import numpy as np

RUNS = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/runs'
RUN = {
    'dvslip': {
        'rif': 'VC_C0_M__20261023_195514_528865__seed42',
        'lineare': 'VC_RI_M--interval-linear__20261026_110201_907982__seed42',
        'simmetrico': 'VC_RI_M--interval-symmetric-r16__20261026_170102_088627__seed42',
        'antisimmetrico': 'VC_RI_M--interval-antisymmetric-r16__20261112_082513_589585__seed42',
        'instradato': 'VC_RI_M--interval-antisymmetric-r16-routed__20261026_105601_490433__seed42',
        'punti': 'VC_RI_M--points-antisymmetric-r16__20261101_021420_408415__seed42',
        'vuoto': 'VC_RI_M--stage2-no-ri__20261028_151520_276458__seed42',
        'mlp': 'VC_RI_M--stage2-matched-mlp__20261028_175955_239379__seed42',
        'gate': 'VC_D_M__20261024_224119_676861__seed42',
        'traccia_statica': 'VC_RT_M--trace-static-antisymmetric-r16__20261101_021457_704527__seed42',
        'traccia_sensore': 'VC_RT_M--trace-sensor-antisymmetric-r16__20261101_102835_087959__seed42',
        'ssa_lineare': 'VC_RI_M--stage2-no-ri-ssa-temporal-linear__20261108_154323_728112__seed42',
        'ssa_condivisa': 'VC_RI_M--stage2-no-ri-ssa-temporal-linrel-antisymmetric__20261108_154222_196226__seed42',
        'ssa_dedicata': 'VC_RI_M--stage2-no-ri-ssa-temporal-linrel-antisymmetric-dedicated__20261108_154250_584407__seed42',
        'stadio1': 'VC_RI_M--interval-antisymmetric-r16-s1__20261110_185130_238825__seed42',
        'stadi12': 'VC_RI_M--interval-antisymmetric-r16-s12__20261110_185035_004274__seed42',
        'consolidamento': 'VC_RC_M--consolidation-functional-k4-interval-antisymmetric-r16__20261113_234837_165646__seed42',
        'remota': 'VC_RI_M--interval-antisymmetric-r16-remote-mean__20261116_020355_999024__seed42',
        'remota_rc': 'VC_RI_M--interval-antisymmetric-r16-remote-rc-k4__20261116_072639_931658__seed42',
        'attivita': 'VC_RI_M--interval-antisymmetric-r16-ct-floor1-active__20261116_012720_934553__seed42',
        'orologio': 'VC_RI_M--interval-antisymmetric-r16-lc-active__20261116_012736_809594__seed42',
        'fbc_rif': 'FBC_C0_M--vc-horizon__20261106_061637_836906__seed42',
        'fbc_irm': 'FBC_RI_M--interval-antisymmetric-r16-vc-horizon__20261106_061549_804495__seed42',
    },
    'thu_eact_50_chl': {
        'rif': 'VC_C0_M__20261023_114802_261612__seed42',
        'lineare': 'VC_RI_M--interval-linear__20261026_153644_696468__seed42',
        'simmetrico': 'VC_RI_M--interval-symmetric-r16__20261026_153644_556166__seed42',
        'antisimmetrico': 'VC_RI_M--interval-antisymmetric-r16__20261026_153642_718466__seed42',
        'instradato': 'VC_RI_M--interval-antisymmetric-r16-routed__20261026_153640_906022__seed42',
        'punti': 'VC_RI_M--points-antisymmetric-r16__20261101_235557_698869__seed42',
        'vuoto': 'VC_RI_M--stage2-no-ri__20261028_220437_545145__seed42',
        'mlp': 'VC_RI_M--stage2-matched-mlp__20261029_005502_913581__seed42',
        'gate': 'VC_D_M__20261025_020803_055847__seed42',
        'traccia_statica': 'VC_RT_M--trace-static-antisymmetric-r16__20261102_000420_752640__seed42',
        'traccia_sensore': 'VC_RT_M--trace-sensor-antisymmetric-r16__20261102_030828_125409__seed42',
        'ssa_lineare': 'VC_RI_M--stage2-no-ri-ssa-temporal-linear__20261108_160142_966758__seed42',
        'ssa_condivisa': 'VC_RI_M--stage2-no-ri-ssa-temporal-linrel-antisymmetric__20261108_173654_554264__seed42',
        'ssa_dedicata': 'VC_RI_M--stage2-no-ri-ssa-temporal-linrel-antisymmetric-dedicated__20261108_160229_702395__seed42',
        'stadio1': 'VC_RI_M--interval-antisymmetric-r16-s1__20261110_190549_925144__seed42',
        'stadi12': 'VC_RI_M--interval-antisymmetric-r16-s12__20261110_191345_091764__seed42',
        'consolidamento': 'VC_RC_M--consolidation-functional-k4-interval-antisymmetric-r16__20261114_104923_287189__seed42',
        'remota': 'VC_RI_M--interval-antisymmetric-r16-remote-mean__20261116_093245_524692__seed42',
        'remota_rc': 'VC_RI_M--interval-antisymmetric-r16-remote-rc-k4__20261116_102424_549297__seed42',
        'attivita': 'VC_RI_M--interval-antisymmetric-r16-ct-floor1-active__20261116_084621_050695__seed42',
        'orologio': 'VC_RI_M--interval-antisymmetric-r16-lc-active__20261116_085649_115472__seed42',
        'fbc_rif': 'FBC_C0_M--vc-horizon__20261106_083202_342388__seed42',
        'fbc_irm': 'FBC_RI_M--interval-antisymmetric-r16-vc-horizon__20261106_094133_652910__seed42',
    },
}


def path(ds, key):
    return os.path.join(RUNS, ds, RUN[ds][key])


def metrics(ds, key):
    d = path(ds, key)
    s = json.load(open(f'{d}/summary.json'))['validation']
    rows = list(csv.DictReader(open(f'{d}/history.csv')))
    col = [k for k in rows[0] if 'macro_f1' in k and 'val' in k][0]
    u16 = 100 * np.mean([float(r[col]) for r in rows[-16:]])
    return dict(f1=100 * s['macro_f1'], acc=100 * s['accuracy'], u16=u16)


def correct(ds, key):
    rows = list(csv.DictReader(open(f'{path(ds, key)}/validation_predictions.csv')))
    return {r['sample_id']: int(r['target']) == int(r['prediction']) for r in rows}


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(ds, first, second):
    """Confronto appaiato: dal primo al secondo modello."""
    p, q = correct(ds, first), correct(ds, second)
    ids = sorted(p); assert set(ids) == set(q)
    fixed = sum(q[i] and not p[i] for i in ids)      # corretti dal secondo
    broken = sum(p[i] and not q[i] for i in ids)     # introdotti dal secondo
    dacc = 100 * (fixed - broken) / len(ids)
    return dict(dacc=dacc, fixed=fixed, broken=broken, p=mcnemar(fixed, broken), n=len(ids))


if __name__ == '__main__':
    for ds in RUN:
        print('=====', ds)
        for k in RUN[ds]:
            m = metrics(ds, k)
            print(f"  {k:15s} F1 {m['f1']:6.2f}  u16 {m['u16']:6.2f}  acc {m['acc']:6.2f}")
        pairs = [('rif', k) for k in RUN[ds] if k != 'rif'] + [
            ('lineare', 'antisimmetrico'), ('simmetrico', 'antisimmetrico'), ('instradato', 'antisimmetrico'),
            ('punti', 'antisimmetrico'), ('vuoto', 'antisimmetrico'), ('mlp', 'antisimmetrico'),
            ('gate', 'antisimmetrico'), ('lineare', 'simmetrico'),
            ('traccia_statica', 'traccia_sensore'), ('traccia_sensore', 'antisimmetrico'), ('traccia_statica', 'antisimmetrico'),
            ('vuoto', 'ssa_lineare'), ('vuoto', 'ssa_condivisa'), ('ssa_lineare', 'ssa_condivisa'), ('ssa_condivisa', 'antisimmetrico'),
            ('ssa_dedicata', 'antisimmetrico'), ('stadio1', 'antisimmetrico'), ('stadi12', 'antisimmetrico'),
            ('consolidamento', 'antisimmetrico'), ('lineare', 'traccia_statica'),
            ('remota', 'antisimmetrico'), ('remota_rc', 'antisimmetrico'), ('attivita', 'antisimmetrico'),
            ('orologio', 'antisimmetrico'), ('fbc_rif', 'fbc_irm'), ('fbc_irm', 'antisimmetrico'), ('fbc_rif', 'rif')]
        for a, b in pairs:
            r = paired(ds, a, b)
            print(f"  {a:>14s} -> {b:15s} dacc {r['dacc']:+6.2f}  {r['fixed']:4d}/{r['broken']:<4d} p={r['p']:.2g}")
