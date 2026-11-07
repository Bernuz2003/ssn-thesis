"""Event frame, voxel grid e time surface sullo stesso intervallo di un campione reale di DVS128 Gesture."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from common1 import *

z = np.load(GESTURE)
t = z['t'].astype(np.int64); x = z['x'].astype(int); y = z['y'].astype(int); p = z['p'].astype(int)
H = W = 128
T0, T1 = 2_000_000, 2_100_000          # finestra di 100 ms
B, TAU = 5, 50_000                     # bin della voxel grid, costante della time surface [us]
m = (t >= T0) & (t < T1)
tw, xw, yw, pw = t[m], x[m], y[m], p[m]
print('eventi nella finestra', m.sum())

# (a) event frame a due canali
on = np.zeros((H, W)); off = np.zeros((H, W))
np.add.at(on, (yw[pw == 1], xw[pw == 1]), 1); np.add.at(off, (yw[pw == 0], xw[pw == 0]), 1)
# (b) voxel grid con interpolazione lineare nel tempo (Zhu et al.)
ts = (B - 1) * (tw - tw.min()) / (tw.max() - tw.min())
sgn = np.where(pw == 1, 1.0, -1.0)
V = np.zeros((B, H, W))
for b in range(B):
    w = np.maximum(0, 1 - np.abs(b - ts))
    np.add.at(V[b], (yw, xw), sgn * w)
# (c) time surface dell'ultimo evento, valutata alla fine della finestra
mh = t < T1
last = np.full((2, H, W), -np.inf)
np.maximum.at(last, (p[mh], y[mh], x[mh]), t[mh].astype(float))
S = np.where(np.isfinite(last), np.exp(-(T1 - last) / TAU), 0).max(0)

red = LinearSegmentedColormap.from_list('r', ['#ffffff', '#f6c3c2', ON, '#8f1d1c'])
blue = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', OFF, '#0d366b'])
div = LinearSegmentedColormap.from_list('d', ['#0d366b', OFF, MID, ON, '#8f1d1c'])
seq = LinearSegmentedColormap.from_list('s', ['#ffffff', '#cde2fb', '#5598e7', '#184f95', '#0d0d0d'])

fig = plt.figure(figsize=(WIDTH, 3.45))
gs = fig.add_gridspec(2, 15, height_ratios=[1.0, 0.62], hspace=0.12, wspace=0.6)
vmax = float(np.ceil(np.percentile(np.concatenate([on[on > 0], off[off > 0]]), 98)))
panels = [(gs[0, 0:5], on, red, 0, vmax, r"\textbf{(a)} event frame, ON"),
          (gs[0, 5:10], off, blue, 0, vmax, r"\textbf{(a)} event frame, OFF"),
          (gs[0, 10:15], S, seq, 0, 1, r"\textbf{(c)} time surface, $\tau_s=50$\,ms")]
for spec, img, cmap, lo, hi, title in panels:
    ax = fig.add_subplot(spec)
    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title, loc='left', fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_color(GRID)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.ax.yaxis.set_major_formatter(comma(1 if hi <= 1 else 0))
vv = np.percentile(np.abs(V[V != 0]), 98)
for b in range(B):
    ax = fig.add_subplot(gs[1, 3 * b:3 * b + 3])
    ax.imshow(V[b], cmap=div, norm=TwoSlopeNorm(0, -vv, vv), interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(True); s_.set_color(GRID)
    ax.set_xlabel(r"$b=%d$" % b, fontsize=7.5, labelpad=2)
    if b == 0:
        ax.set_title(r"\textbf{(b)} voxel grid con $B=5$ bin temporali (rosso positivo, blu negativo)", loc='left', fontsize=8.5)
save(fig, 'cap1_rappresentazioni')
