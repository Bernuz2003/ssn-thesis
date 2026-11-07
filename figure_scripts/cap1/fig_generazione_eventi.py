"""Modello ideale del pixel DVS su un segnale sintetico (illustrativo)."""
import numpy as np
from common1 import *

C = 0.2
t = np.linspace(0, 1, 4001)
sig = lambda x: 1 / (1 + np.exp(-x))
win = np.clip((t - 0.56) / 0.14, 0, 1); win = np.sin(np.pi * win) ** 2
L = (0.9 * sig((t - 0.18) / 0.025) - 0.95 * sig((t - 0.42) / 0.03)
     - 0.62 * np.exp(-((t - 0.85) / 0.04) ** 2)
     + 0.09 * np.sin(2 * np.pi * 21 * (t - 0.56)) * win)
ref = L[0]; refs = np.empty_like(L); ev = []
for i, (ti, li) in enumerate(zip(t, L)):
    while li - ref >= C:
        ref += C; ev.append((ti, ref, +1))
    while li - ref <= -C:
        ref -= C; ev.append((ti, ref, -1))
    refs[i] = ref
print(len(ev), 'eventi:', sum(e[2] > 0 for e in ev), 'ON', sum(e[2] < 0 for e in ev), 'OFF')

fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 2.7), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.8], hspace=0.12))
a1.fill_between(t, refs - C, refs + C, step='post', color=GRID, alpha=0.9, linewidth=0, label=r'fascia $\pm C$ attorno al riferimento')
a1.plot(t, refs, color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), drawstyle='steps-post', label='valore di riferimento')
a1.plot(t, L, color=INK, linewidth=1.5, label=r'$L(t)=\log I(t)$')
for ti, li, p in ev:
    a1.plot(ti, li, marker='^' if p > 0 else 'v', markersize=5.5, color=ON if p > 0 else OFF, markeredgecolor='white', markeredgewidth=0.5, zorder=5)
a1.annotate('oscillazione sotto soglia,\nnessun evento', xy=(0.63, 0.1), xytext=(0.64, 0.62),
            fontsize=7.5, color=INK2, ha='center', arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.6))
a1.set_ylabel(r'$L$ rispetto all\textquotesingle inizio')
a1.yaxis.set_major_formatter(comma(1))
a1.legend(loc='lower left', fontsize=7.2, handlelength=1.6)
a1.set_ylim(-0.95, 1.12)
for ti, li, p in ev:
    a2.plot([ti, ti], [0, p], color=ON if p > 0 else OFF, linewidth=1.4, solid_capstyle='round')
a2.axhline(0, color=MUTED, linewidth=0.6)
a2.set_ylim(-1.4, 1.4); a2.set_yticks([-1, 1]); a2.set_yticklabels(['$-1$', '$+1$'])
a2.set_ylabel('$p_i$')
a2.grid(axis='y', visible=False)
a2.set_xlabel('tempo [s]')
a2.xaxis.set_major_formatter(comma(1))
a2.set_xlim(0, 1)
save(fig, 'cap1_generazione_eventi')
