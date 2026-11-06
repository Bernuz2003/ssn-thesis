import sys, numpy as np
sys.path.insert(0, '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/src')
sys.path.insert(0, '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/notebooks')
from etsr.evaluation.mechanistic.encoding_audit import parse_partition
from mechanistic_plots import centroids
from style import *

z = np.load('/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/data/DvsGesture/events/train/6/user17_led__06.npz')
t, x, y = z['t'].astype(np.int64), z['x'].astype(float), z['y'].astype(float)
centre = (x.mean(), y.mean())
VC_US = 164358
def vc_partition(t):
    t0 = t[0]; T = int(np.ceil((t[-1] - t0 + 1) / VC_US))
    edges = t0 + np.arange(T + 1, dtype=np.float64) * VC_US
    return (t - t0) // VC_US, T, np.stack((edges[:-1], edges[1:]), axis=1)
specs = [("ref", "Riferimento, 20\\,ms", parse_partition("physical:20:20000", {}), NEUTRAL, None),
         ("E0", CLOCK['E0']['label'], parse_partition("physical:500:20000", {}), CLOCK['E0']['color'], 'o'),
         ("FEC", CLOCK['FEC']['label'], parse_partition("events:40", {}), CLOCK['FEC']['color'], 's'),
         ("VC", CLOCK['VC']['label'], vc_partition, CLOCK['VC']['color'], '^')]
tracks = {}
for key, label, part, colour, mk in specs:
    index, frames, bounds = part(t)
    cx, cy, count = centroids(index, frames, x, y)
    valid = count >= 30
    angle = np.unwrap(np.arctan2(cy[valid] - centre[1], cx[valid] - centre[0]))
    times = bounds[valid].mean(1) / 1e6
    tracks[key] = (times, angle, colour, label, mk)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.3))
for key in ("ref", "E0", "FEC", "VC"):
    times, angle, colour, label, mk = tracks[key]
    a1.plot(times, (angle - angle[0]) / (2 * np.pi), color=colour, linewidth=2.2 if key == 'ref' else 1.5,
            marker=mk, markersize=3.2 if mk else 0, markevery=1 if key == 'E0' else 4, label=label, zorder=2 if key == 'ref' else 3)
a1.axhline(0, color=MUTED, linewidth=0.6)
a1.set_xlabel("tempo dall'inizio del segmento [s]")
a1.set_ylabel("giri ricostruiti")
a1.xaxis.set_major_formatter(comma(0)); a1.yaxis.set_major_formatter(comma(0))
a1.set_title(r"\textbf{(a)} rotazione ricostruita dai baricentri", loc='left')
a1.legend(loc='upper left', handlelength=1.6, fontsize=7.5, borderaxespad=0.1)

fine_time, fine_angle = tracks['ref'][0], tracks['ref'][1]
stats = {}
for pos, key in enumerate(("E0", "FEC", "VC")):
    times, _, colour, label, mk = tracks[key]
    times = times[(times >= fine_time[0]) & (times <= fine_time[-1])]
    steps = np.abs(np.diff(np.interp(times, fine_time, fine_angle))) / np.pi
    stats[key] = (float(np.median(steps)), int((steps >= 1).sum()), len(steps))
    jit = np.linspace(-0.16, 0.16, len(steps))
    a2.scatter(pos + jit, steps, s=9, color=colour, alpha=0.75, linewidths=0, marker=mk)
    med = np.median(steps)
    a2.plot([pos - 0.24, pos + 0.24], [med, med], color=INK, linewidth=1.8, solid_capstyle='round')
    a2.text(pos, 2.12, r"$%d/%d$" % ((steps >= 1).sum(), len(steps)), ha='center', fontsize=7.5, color=INK2)
a2.axhline(1, color=INK2, linewidth=0.9, linestyle=(0, (4, 2)))
a2.text(2.45, 1.0, r"$\pi$", va='center', ha='left', fontsize=8.5, color=INK2)
a2.set_xlim(-0.5, 2.5); a2.set_ylim(0, 2.3)
a2.set_xticks([0, 1, 2]); a2.set_xticklabels(["Finestra\nfissa", "Eventi\ncostanti", "Lunghezza\nvariabile"])
a2.set_yticks([0, 0.5, 1, 1.5, 2]); a2.set_yticklabels(["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
a2.set_ylabel(r"$|\Delta\theta|$ tra bin consecutivi")
a2.set_title(r"\textbf{(b)} rotazione tra due passi", loc='left')
a2.grid(axis='x', visible=False)
print(stats)
save(fig, 'cap4_aliasing')
