"""Errori per durata dei campioni (sezione 6.4).

(a) DVS-Lip, validazione di sviluppo: accuratezza per quartile di durata (passi validi del modello di riferimento)
    per modello di riferimento, IRM antisimmetrico, IRM a 2 milioni di parametri e studente della distillazione;
(b) THU-EACT-50-CHL: guadagno di accuratezza dell'IRM sul modello di riferimento per quartile, sulla validazione di
    sviluppo (risultati6.quartili) e sul test ufficiale (artifacts/reports/c0_ri_diagnostics/csv/E1.csv, quartili
    calcolati sul test). Il checkpoint IRM di THU e' lo stesso nelle due partizioni.
"""
import csv
import numpy as np
from matplotlib.lines import Line2D
from common6 import *
from risultati6 import quartili

E1 = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/artifacts/reports/c0_ri_diagnostics/csv/E1.csv'
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw=dict(wspace=0.3))

ax = axes[0]
MOD = [('rif', 'riferimento', NEUTRAL, 'o'), ('antisimmetrico', 'IRM', "#2a78d6", 's'),
       ('irm_2m', 'IRM, 2 M', "#0d3d78", '^'), ('kd5', 'distillazione', "#1baf7a", 'D')]
edges, n, acc = quartili('dvslip', [m[0] for m in MOD])
x = np.arange(1, 5)
for key, lab, col, mk in MOD:
    ax.plot(x, acc[key], color=col, marker=mk, ms=4.5, lw=1.5, label=lab, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([r"Q%d" % i for i in x])
ax.set_xlabel("quartile di durata, dal più breve", fontsize=7.5)
ax.set_ylabel(r"accuratezza [\%]", fontsize=7.5)
ax.set_ylim(45, 78)
ax.legend(loc='upper left', fontsize=6.8, ncol=2, handlelength=1.6, columnspacing=1.0)
ax.set_title(r"\textbf{(a)} DVS-Lip, validazione", loc='left', fontsize=8.2)

ax = axes[1]
_, _, accv = quartili('thu_eact_50_chl', ['rif', 'antisimmetrico'])
gv = np.array(accv['antisimmetrico']) - np.array(accv['rif'])
rows = list(csv.DictReader(open(E1)))
t = {(r['modello'], r['bucket']): float(r['accuracy']) for r in rows if r['dataset'] == 'thu_eact_50_chl'}
gt = np.array([t[('B-', f'Q{i}')] - t[('C0', f'Q{i}')] for i in x])
ax.bar(x - 0.19, gv, width=0.36, color="#7fb0ea", label='validazione', zorder=2)
ax.bar(x + 0.19, gt, width=0.36, color="#0d3d78", label='test ufficiale', zorder=2)
for xi, a, b in zip(x, gv, gt):
    ax.text(xi - 0.19, a + 0.4, r"$%s$" % num(a, 1), ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.text(xi + 0.19, b + 0.4, r"$%s$" % num(b, 1), ha='center', va='bottom', fontsize=6.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([r"Q%d" % i for i in x])
ax.set_xlabel("quartile di durata, dal più breve", fontsize=7.5)
ax.set_ylabel("guadagno dell'IRM [punti]", fontsize=7.5)
ax.set_ylim(0, 19.5)
ax.set_yticks([0, 5, 10, 15])
ax.yaxis.set_major_formatter(comma(0))
ax.grid(axis='x', visible=False); ax.tick_params(axis='x', length=0)
ax.legend(loc='upper center', fontsize=6.8, ncol=2)
ax.set_title(r"\textbf{(b)} THU-EACT-50-CHL", loc='left', fontsize=8.2)
save(fig, 'cap6_durata_vecchia')  # superata da fig_errori_lip.py (2026-11-19)
print('Lip edges', edges, n, {k: np.round(v, 1) for k, v in acc.items()})
print('THU gain val', np.round(gv, 2), 'test', np.round(gt, 2))
