"""Relazione antisimmetrica e simmetrica su un esempio sintetico (sezione 5.3).

Per una coppia di proiezioni (a, b) si definisce c(x) = a.x + i b.x. Allora, per ogni componente,
  rho^- = (a.m)(b.z) - (b.m)(a.z) = |c_m||c_z| sin(theta_z - theta_m),
  rho^+ = (a.m)(b.z) + (b.m)(a.z) = |c_m||c_z| sin(theta_z + theta_m).
(a) lettura geometrica; (b) un vettore di modulo unitario ruota in un verso, si ferma e ruota nel verso opposto;
il passato e' il presente di d = 3 passi prima. Nessun dato sperimentale.
"""
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch
from common5 import *

ANTI, SYM = "#2a78d6", "#eb6834"

fig = plt.figure(figsize=(WIDTH, 2.75))
gs = fig.add_gridspec(2, 2, width_ratios=[0.82, 1.6], height_ratios=[0.8, 1.25], wspace=0.22, hspace=0.18)

# (a) lettura geometrica
ax = fig.add_subplot(gs[:, 0])
t = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(t), np.sin(t), color=GRID, lw=1.0)
ax.plot([-1.12, 1.12], [0, 0], color=GRID, lw=0.8); ax.plot([0, 0], [-1.12, 1.12], color=GRID, lw=0.8)
tm, tz = np.deg2rad(20), np.deg2rad(78)
for th, col, lab, dx, dy in ((tm, INK2, r"$c(m)$" "\n" "passato", 0.05, -0.05), (tz, INK, r"$c(z)$" "\n" "presente", 0.12, -0.16)):
    ax.add_patch(FancyArrowPatch((0, 0), (np.cos(th), np.sin(th)), arrowstyle='-|>', mutation_scale=9, lw=1.4,
                                 color=col, shrinkA=0, shrinkB=0))
    ax.text(np.cos(th) + dx, np.sin(th) + dy, lab, fontsize=7.2, color=col)
ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=20, theta2=78, color=ANTI, lw=1.3))
ax.text(0.36 * np.cos(np.deg2rad(49)) + 0.02, 0.36 * np.sin(np.deg2rad(49)) + 0.02, r"$\Delta\theta$", fontsize=7.5, color=ANTI)
ax.text(-1.05, -1.32, r"$\rho^-\propto\sin(\theta_z-\theta_m)$", fontsize=7.6, color=ANTI)
ax.text(-1.05, -1.6, r"$\rho^+\propto\sin(\theta_z+\theta_m)$", fontsize=7.6, color=SYM)
ax.set_xlim(-1.15, 1.55); ax.set_ylim(-1.7, 1.25); ax.set_aspect('equal'); ax.axis('off')
ax.text(1.0, -0.2, r"$a^{\top}x$", fontsize=7.2, color=MUTED)
ax.text(-0.36, 1.03, r"$b^{\top}x$", fontsize=7.2, color=MUTED)
ax.set_title(r"\textbf{(a)} lettura geometrica", loc='left', fontsize=8.5)

# (b) risposta nel tempo
T, d, w = 100, 3, 0.25
th = np.zeros(T)
for k in range(1, T):
    step = w if k < 40 else (0.0 if k < 60 else -w)
    th[k] = th[k - 1] + step
past = np.concatenate([np.full(d, th[0]), th[:-d]])
anti, sym = np.sin(th - past), np.sin(th + past)
x = np.arange(T)

a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)
for a in (a1, a2):
    for lo, hi, col in ((0, 40, '#f3f6fb'), (60, 100, '#f3f6fb')):
        a.axvspan(lo, hi, color=col, zorder=0, lw=0)
a1.plot(x, th / (2 * np.pi), color=INK, lw=1.4)
a1.set_ylabel("giri", fontsize=7.5); a1.yaxis.set_major_formatter(comma(0))
a1.tick_params(labelbottom=False); a1.grid(axis='x', visible=False)
for xc, lab in ((20, "rotazione in un verso"), (50, "pausa"), (80, "verso opposto")):
    a1.text(xc, 1.95, lab, ha='center', va='top', fontsize=7, color=INK2)
a1.set_ylim(0, 2.05); a1.set_yticks([0, 1])
a1.set_title(r"\textbf{(b)} un vettore che ruota, si ferma e torna indietro", loc='left', fontsize=8.5)

a2.plot(x, sym, color=SYM, lw=1.3, label=r"simmetrica $\rho^+$")
a2.plot(x, anti, color=ANTI, lw=1.8, label=r"antisimmetrica $\rho^-$")
a2.axhline(0, color=MUTED, lw=0.6)
a2.set_ylim(-1.15, 1.15); a2.set_yticks([-1, 0, 1]); a2.yaxis.set_major_formatter(comma(0))
a2.set_xlim(0, T - 1); a2.set_xlabel("passo", fontsize=7.5)
a2.grid(axis='x', visible=False)
a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=7, ncol=2, handlelength=1.6)
save(fig, 'cap5_relazione')
