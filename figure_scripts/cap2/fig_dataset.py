"""Un campione per dataset: tre frame di conteggio e frequenza degli eventi sulla finestra fissa.

Campioni delle partizioni di sviluppo, scelti con durata vicina alla mediana del dataset.
DVS-Lip e DVS128 Gesture dal repository di ricerca; CIFAR10-DVS e THU-EACT-50-CHL copiati
dal repository condiviso SMILIES (data/...) nella cartella dati/, esclusa da git.
I frame sono i bin della rappresentazione a finestra fissa (Delta t del dataset), mostrati come
differenza ON - OFF.
"""
import sys
from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from common2 import *  # noqa

sys.path.insert(0, RESEARCH + '/src')
from etsr.cifar10dvs.dataset import load_cifar_events  # noqa: E402
from etsr.thueact50chl.dataset import load_chl_events  # noqa: E402

HERE = Path(__file__).resolve().parent


def lip():
    a = np.load(RESEARCH + '/data/DVS-Lip/DVS-Lip/train/warning/128.npy', allow_pickle=False)
    return a['t'], a['x'], a['y'], a['p']


def gesture():
    z = np.load(RESEARCH + '/data/DvsGesture/events/train/1/user07_fluorescent__01.npz')
    return z['t'], z['x'], z['y'], z['p']


def cifar():
    e = load_cifar_events(HERE / 'dati/CIFAR10-DVS/cifar10_airplane_318.aedat')
    return e['t'], e['x'], e['y'], e['p']


def thu():
    e = load_chl_events(HERE / 'dati/THU-EACT-50-CHL/A39P17C2-2021_11_06_15_09_46.npy')
    return e['t'], e['x'], e['y'], e['p']


# nome, lettore, finestra [s], bin [s], risoluzione (H, W), bin della curva di frequenza [s]
DATASETS = [
    ("DVS-Lip", lip, 2.0, 0.05, (128, 128), 0.02),
    ("DVS128 Gesture", gesture, 20.0, 0.5, (128, 128), 0.1),
    ("CIFAR10-DVS", cifar, 2.0, 0.05, (128, 128), 0.02),
    ("THU-EACT-50-CHL", thu, 8.0, 0.2, (260, 346), 0.05),
]
FRAC = (0.2, 0.5, 0.8)    # posizione dei tre frame lungo la registrazione

div = LinearSegmentedColormap.from_list('d', ['#0d366b', OFF, MID, ON, '#8f1d1c'])
BAND = '#e9e8e4'

fig = plt.figure(figsize=(WIDTH, 5.15))
gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 0.12, 2.15, 0.02], hspace=0.62, wspace=0.08)

for r, (name, reader, win, dt, (H, W), rbin) in enumerate(DATASETS):
    t, x, y, p = reader()
    t = (np.asarray(t, dtype=np.int64) - 0) / 1e6          # secondi dall'origine del campione
    x = np.asarray(x).astype(int); y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    dur = t.max()
    print(f"{name}: {len(t)} eventi, durata {dur:.2f} s, finestra {win} s, occupata {dur / win:.0%}")

    ks = [int(np.floor(f * dur / dt)) for f in FRAC]
    for c, k in enumerate(ks):
        m = (t >= k * dt) & (t < (k + 1) * dt)
        img = np.zeros((H, W))
        np.add.at(img, (y[m & (p == 1)], x[m & (p == 1)]), 1)
        np.add.at(img, (y[m & (p == 0)], x[m & (p == 0)]), -1)
        v = max(np.percentile(np.abs(img[img != 0]), 97), 1) if (img != 0).any() else 1
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img, cmap=div, norm=TwoSlopeNorm(0, -v, v), interpolation='nearest', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_color(GRID)
        ax.set_xlabel(rf"$k={k}$", fontsize=7.5, labelpad=2)
        if c == 0:
            ax.set_title(name, loc='left', fontsize=8.5, pad=4)

    ax = fig.add_subplot(gs[r, 4])
    edges = np.arange(0, win + rbin, rbin)
    h, _ = np.histogram(t, bins=edges)
    rate = h / rbin / 1e3
    ax.axvspan(dur, win, color=BAND, lw=0, zorder=0)
    for k in ks:
        ax.axvspan(k * dt, (k + 1) * dt, color='#c9c8c3', lw=0, zorder=1, alpha=0.9)
    ax.plot(edges[:-1] + rbin / 2, rate, color=INK, lw=0.9, zorder=3)
    ax.set_xlim(0, win)
    ax.set_ylim(0, rate.max() * 1.12)
    ax.yaxis.set_major_formatter(comma(0))
    ax.tick_params(labelsize=7)
    if r == 0:
        ax.set_title(r"migliaia di eventi al secondo", loc='left', fontsize=8, pad=4, color=INK2)
    ax.xaxis.set_major_formatter(comma(0 if win >= 5 else 1))
    if dur < win * 0.9:
        ax.text((dur + win) / 2, rate.max() * 0.95, "coda vuota", ha='center', va='top', fontsize=7, color=INK2)
    if r == len(DATASETS) - 1:
        ax.set_xlabel(r"tempo dall'origine del campione [s]", fontsize=7.5)

save(fig, "cap2_dataset")
