"""Tabelle estese dell'appendice B, generate dagli artifact del repository di ricerca.

Scrive in content/appendice/ e inserisce le righe in content/appendix.tex, tra i marcatori INIZIO/FINE:
  confronti.tex    confronti appaiati di ogni variante dei capitoli 5 e 6 con l'IRM antisimmetrico compatto
  compressione.tex sweep di compressione per rango (sola inferenza)
  coppie.tex       scambi dentro ciascuna delle 25 coppie di parole simili di DVS-Lip
Uso: python tabelle_appendice.py (dalla cartella figure_scripts/appendice).
Nota: su DVS-Lip lo sweep usa per ora il checkpoint IRM 20261026_112001 (solo differenze dal proprio denso).
"""
import json, os, sys
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
    s = s.replace('.', '{,}')
    return f"${s}$"


def pval(p):
    if p >= 0.001:
        return f"${p:.3f}$".replace('.', '{,}')
    m, e = f"{p:.1e}".split('e')
    return f"${m.replace('.', '{,}')}\\cdot10^{{{int(e)}}}$"


# ---- confronti appaiati ---------------------------------------------------------------------------------------
ROWS = [
    ('Capitolo 5, forme della relazione e controlli', None),
    ('Modello di riferimento', 'rif'), ('IRM lineare', 'lineare'), ('IRM simmetrico', 'simmetrico'),
    ('IRM con instradamento', 'instradato'), ('Sito vuoto', 'vuoto'), ('MLP di pari parametri', 'mlp'),
    ('Relazione a punti', 'punti'), ('Gate di contenuto', 'gate'),
    ('Capitolo 5, alternative', None),
    ('Tracce a decadimento costante', 'traccia_statica'), ('Tracce guidate dagli eventi', 'traccia_sensore'),
    ('Memoria di consolidamento', 'consolidamento'), ('Bande e memoria remota', 'remota'),
    ('SSA, solo richiamo', 'ssa_lineare'), ('SSA, confronto su Key e Value', 'ssa_condivisa'),
    ('SSA, proiezioni dedicate', 'ssa_dedicata'), ('Primo patch embedding', 'stadio1'),
    ('Entrambi i patch embedding', 'stadi12'), ('Bande in tempo di attività', 'attivita'),
    ('Orologio appreso', 'orologio'),
    ('Capitolo 6', None),
    ('IRM, 1 milione di parametri', 'irm_1m'), ('IRM, 2 milioni di parametri', 'irm_2m'),
    ('IRM con distillazione', 'kd5'), ('IRM con auto-ensemble', 'irm_se'),
]
lines = []
for lab, key in ROWS:
    if key is None:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\")
        continue
    cells = [f"\\quad {lab}"]
    for ds in DS:
        x = r6.paired(ds, 'antisimmetrico', key)
        d = r6.metrics(ds, key)['f1'] - r6.metrics(ds, 'antisimmetrico')['f1']
        cells += [n(d, 1, True), f"${x['fixed']}/{x['broken']}$", pval(x['p'])]
    lines.append(' & '.join(cells) + ' \\\\')
open(f'{OUT}/confronti.tex', 'w').write('\n'.join(lines) + '\n')

# ---- compressione -------------------------------------------------------------------------------------------
sw = json.load(open(f'{REP}/b_minus_compression/rank_sweep.json'))
POS = {'dvslip': 64, 'thu_eact_50_chl': 374}
lines = []
for ds, name in [('dvslip', 'DVS-Lip'), ('thu_eact_50_chl', 'THU-EACT-50-CHL')]:
    lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
    dense = sw[f'{ds}/dense']
    keys = [f'{ds}/dense'] + sorted((k for k in sw if k.startswith(ds + '/q')), key=lambda k: sw[k]['rank'])
    for k in keys:
        r = sw[k]
        q = r.get('rank')
        lab = 'denso' if k.endswith('dense') else f"$q={q}$"
        ch = 128 if q is None else q
        state = 23 if ds != 'dvslip' else 15
        st = POS[ds] * state * (ch + 32) / 1e3
        # MAC del sito per passo: (K+1)Cq + (K+2)Cr per posizione (compresso), KC^2 + (K+2)Cr (denso)
        per_pos = (4 * 128 * 128 if q is None else 5 * 128 * q) + 6 * 128 * 16
        mac = POS[ds] * per_pos / 1e6
        en = 100 * (r.get('retained_weight_energy') or 1.0)
        df = 100 * (r['f1'] - dense['f1'])
        par = f"{r['params_s2_mixer_all']:,}".replace(',', '\\,')
        stt = f"{st:,.0f}".replace(',', '\\,')
        lines.append(f"\\quad {lab} & {n(en, 1)} & {n(df, 1, True) if q else '--'} & ${par}$ & {n(mac, 2)} & ${stt}$ \\\\")
open(f'{OUT}/compressione.tex', 'w').write('\n'.join(lines) + '\n')

# ---- coppie di parole simili --------------------------------------------------------------------------------
MOD = ['rif', 'antisimmetrico', 'irm_2m', 'kd5']
pairs = {}
for k in MOD:
    v = json.load(open(f"{r6.path('dvslip', k)}/summary.json"))['validation']['class_group_accuracies']
    for p in v['visually_confusable_pair_errors']['pairs']:
        pairs.setdefault(tuple(p['classes']), {})[k] = (p['within_pair_errors'], p['samples'])
order = sorted(pairs, key=lambda c: -pairs[c]['antisimmetrico'][0])
lines = []
for c in order:
    row = [f"\\emph{{{c[0]}}} / \\emph{{{c[1]}}}", f"${pairs[c]['rif'][1]}$"] + [f"${pairs[c][k][0]}$" for k in MOD]
    lines.append(' & '.join(row) + ' \\\\')
tot = [sum(pairs[c][k][0] for c in pairs) for k in MOD]
lines.append('\\hline')
lines.append('Totale & $' + f"{sum(pairs[c]['rif'][1] for c in pairs):,}".replace(',', '\\,') + '$ & ' + ' & '.join(f'${t}$' for t in tot) + ' \\\\')
open(f'{OUT}/coppie.tex', 'w').write('\n'.join(lines) + '\n')
print('ok', tot)

# ---- inserimento nell'appendice, tra i marcatori -------------------------------------------------------------
import re
APP = os.path.join(HERE, '..', '..', 'content', 'appendix.tex')
text = open(APP).read()
for name in ['confronti', 'compressione', 'coppie']:
    body = open(f'{OUT}/{name}.tex').read()
    pat = re.compile(r'(% INIZIO ' + name + r'[^\n]*\n).*?(% FINE ' + name + r')', re.S)
    assert pat.search(text), name
    text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
open(APP, 'w').write(text)
print('appendice aggiornata')
