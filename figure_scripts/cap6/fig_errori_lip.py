"""Errori residui su DVS-Lip (sezione 6.6): figure cap6_confusione_lip, cap6_errori_percorso e cap6_durata.

cap6_confusione_lip
    Matrici di confusione sul test ufficiale del modello finale di DVS-Lip con due milioni di parametri (IRM, relazioni
    locali prima del pooling e distillazione), su due sottoinsiemi leggibili: (a) le 10 coppie visivamente simili con piu'
    scambi, con le due parole di ciascuna coppia adiacenti; (b) le 20 parole comuni con piu' errori verso altre parole
    comuni, in ordine alfabetico. Colore: numero di clip in scala
    logaritmica, log(1 + conteggio), con la mappa magma del notebook c0_ri_diagnostics del repository di ricerca. Le
    predizioni verso parole dell'altro gruppo non compaiono.
cap6_errori_percorso
    Errori sul test lungo il percorso dei modelli, in percentuale di quelli del modello di riferimento: scambi dentro la
    coppia, altri errori sulle parole delle coppie, errori sulle altre parole.
cap6_durata
    Accuratezza per quartile del numero di passi validi, in validazione (passi dal file delle predizioni del modello di
    riferimento) e sul test (passi da artifacts/reports/c0_ri_diagnostics/workers/dvslip_c0/original), per il modello di
    riferimento e per il modello finale con 0,5, 1 e 2 milioni di parametri.
Configurazione dei gruppi: configs/dvslip_class_groups.json.
"""
import csv, json
import numpy as np
from common6 import *
from finali6 import test
from risultati6 import path

G = json.load(open('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/configs/dvslip_class_groups.json'))
CLS = G['classes']; IDX = {c: i for i, c in enumerate(CLS)}
PAIRS = G['visually_confusable_pairs']
PAIR_WORDS = [w for p in PAIRS for w in p]
COMMON = sorted(G['common_words'])
PARTNER = -np.ones(100, int)
for a, b in PAIRS:
    PARTNER[IDX[a]] = IDX[b]; PARTNER[IDX[b]] = IDX[a]
CONF = PARTNER >= 0
FINAL = 'lip_prepool_2m_kd4'

# ---- matrici di confusione --------------------------------------------------------------------------------------
t = test(FINAL)
C = np.zeros((100, 100))
np.add.at(C, (t['target'], t['pred']), 1)
VMAX = np.log1p(C.max())
# sottoinsiemi leggibili: le 10 coppie con piu' scambi e le 20 parole comuni con piu' errori verso altre parole comuni
swaps = sorted(PAIRS, key=lambda pr: -(C[IDX[pr[0]], IDX[pr[1]]] + C[IDX[pr[1]], IDX[pr[0]]]))
SUB_PAIRS = [w for pr in swaps[:10] for w in pr]
ic = [IDX[w] for w in COMMON]
Cc = C[np.ix_(ic, ic)].copy(); np.fill_diagonal(Cc, 0)
conf_c = Cc.sum(0) + Cc.sum(1)
SUB_COMMON = sorted(COMMON[i] for i in np.argsort(-conf_c, kind='stable')[:20])
print('coppie:', swaps[:10]); print('parole comuni:', SUB_COMMON)
fig = plt.figure(figsize=(WIDTH, 3.0))
axes = [fig.add_axes([0.11, 0.2, 0.36, 0.7]), fig.add_axes([0.58, 0.2, 0.36, 0.7])]
cax = fig.add_axes([0.955, 0.2, 0.012, 0.7])
for ax, words, title in [(axes[0], SUB_PAIRS, r"\textbf{(a)} 10 coppie di parole simili"),
                         (axes[1], SUB_COMMON, r"\textbf{(b)} 20 parole comuni")]:
    ix = [IDX[w] for w in words]
    M = C[np.ix_(ix, ix)]
    im = ax.imshow(np.log1p(M), cmap='magma', vmin=0, vmax=VMAX, interpolation='nearest', aspect='equal')
    n = len(words)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words, rotation=90, fontsize=5.6)
    ax.set_yticklabels(words, fontsize=5.6)
    ax.tick_params(length=0, pad=1)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, loc='left', fontsize=7.8, pad=3)
ticks = [0, 1, 2, 5, 10, 20, 40]
cb = fig.colorbar(im, cax=cax, ticks=np.log1p(ticks))
cb.ax.set_yticklabels([str(v) for v in ticks]); cb.ax.tick_params(labelsize=5.5, length=2)
cb.set_label("clip", fontsize=6.5, labelpad=2)
fig.text(0.5, 0.0, "parola predetta", ha='center', va='bottom', fontsize=7)
fig.text(0.005, 0.54, "parola pronunciata", ha='left', va='center', rotation=90, fontsize=7)
save(fig, 'cap6_confusione_lip')

# ---- errori lungo il percorso ---------------------------------------------------------------------------------
PATH = [("riferimento", 'lip_c0'), ("IRM e relazioni\nlocali", 'lip_prepool'),
        ("+ distillazione", 'lip_prepool_kd4_05'), ("modello finale,\n2 M", FINAL)]
vals = []
for _, w in PATH:
    tt = test(w); y, p = tt['target'], tt['pred']; err = p != y; g1 = CONF[y]
    swap = err & g1 & (p == PARTNER[y])
    vals.append((swap.sum(), (err & g1 & ~swap).sum(), (err & ~g1).sum()))
vals = np.array(vals, float)
print('errori (scambi, altri gruppo 1, gruppo 2):'); print(vals)
rel = 100 * vals / vals[0]
SER = [("scambi dentro la coppia", '#0d3d78', 'o'), ("altri errori, coppie simili", '#2a78d6', 's'),
       ("errori, altre parole", '#9a9993', '^')]
fig, ax = plt.subplots(figsize=(WIDTH * 0.72, 2.45))
x = np.arange(len(PATH))
ax.axhline(100, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
for k, (lab, col, mk) in enumerate(SER):
    ax.plot(x, rel[:, k], '-' + mk, color=col, ms=4.5, lw=1.4, zorder=3, label=lab)
    ax.annotate(r"$%d$" % vals[-1, k], (x[-1], rel[-1, k]), xytext=(7, (0, 4, -4)[k]), textcoords='offset points',
                ha='left', va='center', fontsize=6.3, color=INK2)
    print(lab, np.round(rel[:, k], 1))
ax.set_xticks(x); ax.set_xticklabels([l for l, _ in PATH], fontsize=6.8)
ax.set_xlim(-0.3, len(PATH) - 0.5)
ax.set_ylim(65, 120)
ax.yaxis.set_major_formatter(comma(0, percent=True))
ax.set_ylabel("rispetto al riferimento", fontsize=7.3)
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.legend(loc='lower left', fontsize=6.5, frameon=False)
save(fig, 'cap6_errori_percorso')

# ---- durata ----------------------------------------------------------------------------------------------------
z = np.load('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/c0_ri_diagnostics/'
            'workers/dvslip_c0/original/predictions.npz')
steps_test = dict(zip(z['sample_ids'].tolist(), z['steps'].tolist()))
rows = list(csv.DictReader(open(f"{path('dvslip', 'rif')}/validation_predictions.csv")))
steps_val = {r['sample_id']: int(r['valid_steps']) for r in rows}
DUR = [("riferimento", 'lip_c0', NEUTRAL, 'o'), ("modello finale, 0,5 M", 'lip_prepool_kd4_05', '#9dbfe8', 's'),
       ("modello finale, 1 M", 'lip_prepool_1m_kd4', '#2a78d6', '^'), ("modello finale, 2 M", FINAL, '#0d3d78', 'D')]
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, part, steps, title in [(axes[0], 'val', steps_val, 'validazione di sviluppo'),
                               (axes[1], 'test', steps_test, 'test ufficiale')]:
    ids = test('lip_c0', part=part)['ids'].tolist()
    st = np.array([steps[i] for i in ids]); e = np.quantile(st, [.25, .5, .75])
    b = np.searchsorted(e, st, side='left')
    print(part, 'soglie in passi', e, 'numerosita', np.bincount(b))
    for lab, w, col, mk in DUR:
        tt = test(w, part=part); assert tt['ids'].tolist() == ids
        c = tt['pred'] == tt['target']; q = [100 * c[b == j].mean() for j in range(4)]
        print(f"  {lab:24s}", np.round(q, 1))
        ax.plot(np.arange(4), q, '-' + mk, ms=4.2, lw=1.4, color=col, label=lab,
                mfc='white' if w == 'lip_c0' else col, mec=col, mew=1.1)
    ax.set_xticks(range(4)); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=7.5)
    ax.set_xlabel("quartile di durata, dal più breve", fontsize=7.5)
    ax.set_title(title, fontsize=8.5)
    ax.yaxis.set_major_formatter(comma(0))
axes[0].set_ylabel(r"accuratezza [\%]", fontsize=7.5)
axes[0].legend(loc='upper left', fontsize=6.4, frameon=False)
save(fig, 'cap6_durata')
