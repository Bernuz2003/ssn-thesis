"""Tabelle estese dell'appendice B, generate dagli artifact del repository di ricerca.

Scrive in content/appendice/ e inserisce le righe in content/appendix.tex, tra i marcatori INIZIO/FINE:
  confronti.tex    confronti appaiati di ogni variante dei capitoli 5 e 6 con l'IRM antisimmetrico compatto
  compressione.tex sweep di compressione per rango (sola inferenza)
  coppie.tex       scambi dentro ciascuna delle 25 coppie di parole simili di DVS-Lip
Uso: python tabelle_appendice.py (dalla cartella figure_scripts/appendice).
Nota: su DVS-Lip lo sweep usa per ora il checkpoint IRM 20261026_112001 (solo differenze dal proprio denso).
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cap6'))
sys.path.insert(0, os.path.join(HERE, '..', 'cap5'))
import risultati6 as r6  # noqa: E402

OUT = os.path.join(HERE, '..', '..', 'content', 'appendice')
os.makedirs(OUT, exist_ok=True)
REP = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports'
DS = ['dvslip', 'thu_eact_50_chl']


def n(v, d=1, sign=False):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    if s.lstrip('+-').strip('0.') == '':
        s = s.lstrip('+-')
    s = s.replace('.', '{,}')
    return f"${s}$"


def pval(p):
    if p >= 0.001:
        return f"${p:.3f}$".replace('.', '{,}')
    m, e = f"{p:.1e}".split('e')
    return f"${m.replace('.', '{,}')}\\cdot10^{{{int(e)}}}$"


# ---- confronti appaiati ---------------------------------------------------------------------------------------
ROWS = [
    ('Capitolo 5, controlli della sezione 5.4', None),
    ('Modello di riferimento', 'rif'), ('Sito vuoto', 'vuoto'), ('MLP di pari parametri', 'mlp'),
    ('Bande senza confronto', 'lineare'), ('Confronto simmetrico', 'simmetrico'),
    ('Ritardi puntuali', 'punti'), ('Tracce a decadimento appreso', 'traccia_statica'),
    ('Tracce guidate dal sensore', 'traccia_sensore'), ('Orologio appreso', 'orologio'),
    ('Passato remoto aggiunto', 'remota'),
    ('Capitolo 5, altre prove', None),
    ('Variante condizionale del mixer a ritardi', 'gate'), ('IRM con instradamento per contenuto', 'instradato'),
    ('IRM nel primo patch embedding', 'stadio1'), ('IRM in entrambi i patch embedding', 'stadi12'),
    ('SSA, solo richiamo', 'ssa_lineare'), ('SSA, confronto su Key e Value', 'ssa_condivisa'),
    ('SSA, proiezioni dedicate', 'ssa_dedicata'), ('Memoria di consolidamento', 'consolidamento'),
    ('Bande in tempo di attività', 'attivita'),
    ('Capitolo 6', None),
    ('IRM, 1 milione di parametri', 'irm_1m'), ('IRM, 2 milioni di parametri', 'irm_2m'),
    ('IRM con distillazione', 'kd4'), ('IRM con auto-ensemble', 'irm_se'),
    ('IRM con relazioni locali prima del pooling', 'prepool'),
]
lines = []
for lab, key in ROWS:
    if key is None:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        continue
    cells = [f"\\quad {lab}"]
    for ds in DS:
        if key not in r6.RUN[ds]:          # distillazione: solo DVS-Lip
            cells += ['--', '--', '--']
            continue
        x = r6.paired(ds, 'antisimmetrico', key)
        d = r6.metrics(ds, key)['f1'] - r6.metrics(ds, 'antisimmetrico')['f1']
        cells += [n(d, 1, True), f"${x['fixed']}/{x['broken']}$", pval(x['p'])]
    lines.append(' & '.join(cells) + ' \\\\')
open(f'{OUT}/confronti.tex', 'w').write('\n'.join(lines) + '\n')

# ---- compressione -------------------------------------------------------------------------------------------
# Dal 2026-11-19: report della valutazione finale (artifacts/reports/b_minus_final_inference), tramite
# figure_scripts/cap6/compressione6.py; costi per passo dai profili su 32 clip di validazione.
sys.path.insert(0, os.path.join(HERE, '..', 'cap6'))
import compressione6 as c6
lines = []
for ds, name, w in [('dvslip', 'DVS-Lip', 'lip_b'), ('thu_eact_50_chl', 'THU-EACT-50-CHL', 'thu_b')]:
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
    rep = c6.load(w); o = c6.worker(w)
    for arm in ['dense'] + [f'q{k}' for k in o['ranks']]:
        c = c6.cost(rep['arms'][arm]['profile'])
        q = None if arm == 'dense' else int(arm[1:])
        lab = 'denso' if q is None else (f"$q={q}$" + (' (scelto)' if q == o['q'] else ''))
        en = 100.0 if q is None else o['energy'][q]
        df = '--' if q is None else n(o['val_curve'][q], 1, True)
        par = f"{c['module_params']:,}".replace(',', '\\,')
        stt = f"{c['module_state'] / 1e3:,.0f}".replace(',', '\\,')
        lines.append(f"\\quad {lab} & {n(en, 1)} & {df} & ${par}$ & {n(c['module_mac'] / 1e6, 2)} & ${stt}$ \\\\")
open(f'{OUT}/compressione.tex', 'w').write('\n'.join(lines) + '\n')

ORD = ['primo', 'secondo', 'terzo']
lines = []
for ds, rows in c6.MODELS.items():
    lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{'THU-EACT-50-CHL' if ds != 'dvslip' else 'DVS-Lip'}}}}} \\\\")
    for lab, size, ws in rows:
        for i, w in enumerate(ws):
            o = c6.worker(w)
            name = f"{lab}, {size.replace(' (4 ins.)', ', quattro insegnanti')}" + (f", {ORD[i]} addestramento" if len(ws) > 1 else '')
            if o['q'] is None:
                lines.append(f"\\quad {name} & -- & \\multicolumn{{4}}{{c}}{{nessun rango entro la tolleranza}} & {n(o['test_dense'], 1)} \\\\")
                continue
            d, c = o['dense'], o['comp']
            sv = lambda k: n(-c6.saving(d, c, k), 0)
            lines.append(f"\\quad {name} & ${o['q']}$ & {n(o['energy'][o['q']], 1)} & {sv('module_mac')} & {sv('mac')} & {sv('state')} & "
                         f"{n(o['test_dense'], 1)} $\\to$ {n(o['test_comp'], 1)} \\\\")
open(f'{OUT}/ranghi.tex', 'w').write('\n'.join(lines) + '\n')

# ---- relazioni locali alle tre larghezze ------------------------------------------------------------------------
# Dal 2026-11-20: IRM contro IRM con relazioni locali, con e senza distillazione, sulla validazione e sul test
# (logit di b_minus_final_inference; a 2 M l'IRM e' il primo dei tre addestramenti). McNemar esatto.
import finali6
SCAN = [('0,5 M', 'standard', 'lip_b', 'lip_prepool'), ('0,5 M', 'distillazione', 'lip_kd4_05', 'lip_prepool_kd4_05'),
        ('1 M', 'standard', 'lip_b_1m', 'lip_prepool_1m'), ('1 M', 'distillazione', 'lip_kd4_1m', 'lip_prepool_1m_kd4'),
        ('2 M', 'standard', 'lip_b_2m', 'lip_prepool_2m'), ('2 M', 'distillazione', 'lip_kd4_2m', 'lip_prepool_2m_kd4')]
lines = []
for size, rec, a, b in SCAN:
    cells = [f"\\quad {size}" if rec == 'standard' else '', rec]
    for part in ('val', 'test'):
        A, B = finali6.test(a, part=part), finali6.test(b, part=part)
        ca, cb = A['pred'] == A['target'], B['pred'] == B['target']
        fx, br = int(np.sum(cb & ~ca)), int(np.sum(ca & ~cb))
        cells += [n(A['acc'], 1), n(B['acc'], 1), n(B['acc'] - A['acc'], 1, True), pval(c6.mcnemar(fx, br))]
    lines.append(' & '.join(cells) + ' \\\\')
open(f'{OUT}/scan_locali.tex', 'w').write('\n'.join(lines) + '\n')

# ---- coppie di parole simili --------------------------------------------------------------------------------
# Dal 2026-11-19: test ufficiale, stessi modelli della figura cap6_confusione_lip (media sugli addestramenti ripetuti).
import finali6
Gr = json.load(open('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/configs/dvslip_class_groups.json'))
IDX = {c: i for i, c in enumerate(Gr['classes'])}
PATHM = [['lip_c0'], ['lip_prepool'], ['lip_prepool_kd4_05'], ['lip_prepool_2m_kd4']]
T = {w: finali6.test(w) for ws in PATHM for w in ws}
lines, rows = [], []
for a_, b_ in Gr['visually_confusable_pairs']:
    ia, ib = IDX[a_], IDX[b_]
    y = T['lip_c0']['target']; n_ = int(np.sum((y == ia) | (y == ib)))
    vals = []
    for ws in PATHM:
        v = []
        for w in ws:
            y, p = T[w]['target'], T[w]['pred']
            v.append(int(np.sum((y == ia) & (p == ib)) + np.sum((y == ib) & (p == ia))))
        vals.append(float(np.mean(v)))
    rows.append((a_, b_, n_, vals))
rows.sort(key=lambda r: -r[3][-1])
fmt = lambda v: f"${v:.0f}$" if abs(v - round(v)) < 1e-9 else f"${v:.1f}$".replace('.', '{,}')
for a_, b_, n_, vals in rows:
    lines.append(' & '.join([f"\\emph{{{a_}}} / \\emph{{{b_}}}", f"${n_}$"] + [fmt(v) for v in vals]) + ' \\\\')
tot = [sum(r[3][k] for r in rows) for k in range(len(PATHM))]
lines.append('\\hline')
lines.append('Totale & $' + f"{sum(r[2] for r in rows):,}".replace(',', '\\,') + '$ & ' + ' & '.join(fmt(t) for t in tot) + ' \\\\')
open(f'{OUT}/coppie.tex', 'w').write('\n'.join(lines) + '\n')
print('ok', tot)

# ---- inserimento nell'appendice, tra i marcatori -------------------------------------------------------------
import re
APP = os.path.join(HERE, '..', '..', 'content', 'appendix.tex')
text = open(APP).read()
for name in ['confronti', 'compressione', 'ranghi', 'scan_locali', 'coppie']:
    body = open(f'{OUT}/{name}.tex').read()
    pat = re.compile(r'(% INIZIO ' + name + r'[^\n]*\n).*?(% FINE ' + name + r')', re.S)
    assert pat.search(text), name
    text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
open(APP, 'w').write(text)
print('appendice aggiornata')
