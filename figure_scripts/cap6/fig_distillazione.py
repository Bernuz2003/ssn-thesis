"""Distillazione e auto-ensemble (sezione 6.1): figura cap6_distillazione.

(a) Accuratezza di training (media delle ultime 16 epoche) e Macro-F1 di validazione dell'IRM antisimmetrico compatto,
    addestrato in modo standard, con auto-ensemble e con distillazione da cinque insegnanti (risultati6.py).
(b) Concordanza fra addestramenti ripetuti dell'IRM con due milioni di parametri, con e senza distillazione da quattro
    insegnanti, su validazione e test (media sulle coppie; logit di artifacts/reports/b_minus_final_inference).
    Tre addestramenti (tre coppie) in entrambi i casi.
"""
import itertools
import numpy as np
from matplotlib.patches import FancyArrowPatch
from common6 import *
from risultati6 import metrics, train_u16
from finali6 import test

fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1]))

# ---- (a) -----------------------------------------------------------------------------------------------------
ax = axes[0]
LAB = {'antisimmetrico': 'IRM', 'irm_se': 'auto-ensemble', 'kd5': 'distillazione'}
for ds in ('dvslip', 'thu_eact_50_chl'):
    pts = {k: (train_u16(ds, k), metrics(ds, k)['f1']) for k in LAB}
    col = DSCOL[ds]
    x0, y0 = pts['antisimmetrico']
    for k in ('irm_se', 'kd5'):
        x1, y1 = pts[k]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=1.0,
                                     color=tint(col, 0.75) if k == 'irm_se' else col, shrinkA=4, shrinkB=4, zorder=2))
        ax.plot(x1, y1, 'o', ms=5.5, color='white' if k == 'irm_se' else col, mec=col, mew=1.2, zorder=3)
        dx, ha = (-6, 'right') if ds == 'dvslip' else (6, 'left')
        ax.annotate(LAB[k], (x1, y1), xytext=(dx, 0), textcoords='offset points', ha=ha, va='center',
                    fontsize=6.6, color=INK2)
    ax.plot(x0, y0, '*', ms=9, color=col, mec=INK, mew=0.5, zorder=4, label=DSNAME[ds])
    dx, ha = (-7, 'right') if ds == 'dvslip' else (7, 'left')
    ax.annotate("IRM", (x0, y0), xytext=(dx, 0), textcoords='offset points', ha=ha, va='center',
                fontsize=6.8, color=INK)
ax.set_xlim(72, 101); ax.set_ylim(54, 65)
ax.legend(loc='lower right', fontsize=6.8, frameon=False, handletextpad=0.2, markerscale=0.8)
ax.set_xlabel(r"accuratezza di training [\%]", fontsize=7.5)
ax.set_ylabel(r"Macro-F1 di validazione [\%]", fontsize=7.5)
ax.xaxis.set_major_formatter(comma(0)); ax.yaxis.set_major_formatter(comma(0))
ax.set_title(r"\textbf{(a)} adattamento e validazione", loc='left', fontsize=8.2)

# ---- (b) -----------------------------------------------------------------------------------------------------
def agreement(workers, part):
    P = [test(w, part=part)['pred'] for w in workers]
    return 100 * np.mean([np.mean(a == b) for a, b in itertools.combinations(P, 2)])


ax = axes[1]
GROUPS = [('senza distillazione', ['lip_b_2m', 'lip_b_2m_s43', 'lip_b_2m_s44'], tint(DSCOL['dvslip'], 0.45)),
          ('con distillazione', ['lip_kd4_2m', 'lip_kd4_2m_s43', 'lip_kd4_2m_s44'], DSCOL['dvslip'])]
x = np.arange(2)
for j, (lab, ws, col) in enumerate(GROUPS):
    v = [agreement(ws, p) for p in ('val', 'test')]
    print(lab, np.round(v, 1))
    xs = x + (j - 0.5) * 0.36
    ax.bar(xs, v, width=0.34, color=col, label=lab, zorder=2)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi + 1.0, r"$%s$" % num(vi, 1), ha='center', va='bottom', fontsize=6.8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(['validazione', 'test'], fontsize=7.5)
ax.set_ylim(0, 92)
ax.set_ylabel(r"clip con la stessa predizione [\%]", fontsize=7.5)
ax.yaxis.set_major_formatter(comma(0))
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.legend(loc='upper center', fontsize=6.8, frameon=False, ncol=2, handlelength=1.2, columnspacing=1.0)
ax.set_title(r"\textbf{(b)} DVS-Lip, IRM con 2 M di parametri", loc='left', fontsize=8.2)
save(fig, 'cap6_distillazione')
