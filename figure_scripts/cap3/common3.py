"""Stile comune (da ../cap4), output in images/cap3, lettura dei risultati dal repository di ricerca."""
import collections, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cap4'))
import style
style.OUT = os.path.join(HERE, '..', '..', 'images', 'cap3')
from style import *  # noqa
import numpy as np
from matplotlib.colors import to_rgb

RESEARCH = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research'
# colori dei blocchi, come nella figura 2.2 (palette validata, ordine adiacente verificato)
PE1, ST1, PE2, ST2, HEAD = "#eda100", "#2a78d6", "#1baf7a", "#eb6834", "#9a9993"
BLOCKS = [("patch_embed1", "patch embedding 1", PE1), ("stage1", "stadio 1", ST1),
          ("patch_embed2", "patch embedding 2", PE2), ("stage2", "stadio 2", ST2), ("head", "testa", HEAD)]
# catena incrementale su DVS-Lip, finestra fissa, seed 42 (profili comuni: stessi 64 campioni, FP32 CPU)
CHAIN = [("B", "E0_B_M__20260825_211710__seed42"),
         ("F", "E0_B_M--pyramidal__20260908_094118_261095__seed42"),
         ("TCAP4", "E0_C0_M--f-temporal-capacity-d8__20260913_161320_867938__seed42"),
         ("C0", "E0_C0_M__20260914_093427_434930__seed42")]


def tint(color, a):
    r = np.array(to_rgb(color))
    return tuple(1 - a * (1 - r))


def summary(dataset, run_id):
    return json.load(open(f"{RESEARCH}/artifacts/runs/{dataset}/{run_id}/summary.json"))


def summary_glob(dataset, prefix, seed=42):
    hits = sorted(glob.glob(f"{RESEARCH}/artifacts/runs/{dataset}/{prefix}__*__seed{seed}/summary.json"))
    hits = [h for h in hits if h.split('/')[-2].split('__')[0] == prefix]
    assert len(hits) == 1, (dataset, prefix, seed, hits)
    return json.load(open(hits[0]))


def f1(s):
    return 100 * s["validation"]["macro_f1"]


def profile(run_id):
    return json.load(open(f"{RESEARCH}/artifacts/reports/architecture_evolution/{run_id}/hardware_profile_v4.json"))


def block_totals(p, key):
    agg = collections.Counter()
    for name, e in p["layers"].items():
        if "." not in name and name != "head":
            continue
        if isinstance(e.get(key), (int, float)):
            agg[name.split(".")[0]] += e[key]
    return agg


def save(fig, name):  # noqa: F811
    os.makedirs(style.OUT, exist_ok=True)
    fig.savefig(f"{style.OUT}/{name}.pdf")
    preview = os.path.join(HERE, "anteprime")
    os.makedirs(preview, exist_ok=True)
    fig.savefig(f"{preview}/{name}.png", dpi=170)
    plt.close(fig)
