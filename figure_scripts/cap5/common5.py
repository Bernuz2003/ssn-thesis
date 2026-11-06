"""Riusa stile e loader del Capitolo 4, con output in images/cap5."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cap4'))
import style
style.OUT = os.path.join(HERE, '..', '..', 'images', 'cap5')
from style import *  # noqa
from load import DS, load  # noqa

# Ritardi del mixer a ritardi riscalati per la rappresentazione a lunghezza variabile
DELAYS = {'dvslip': [2, 4, 7, 15], 'dvsgesture': [3, 6, 12, 24],
          'cifar10dvs': [2, 3, 6, 12], 'thu_eact_50_chl': [3, 6, 11, 23]}
DELTA_MS = {'dvslip': 27.422, 'dvsgesture': 164.358, 'cifar10dvs': 32.877, 'thu_eact_50_chl': 69.995}
# Rampa sequenziale a una tinta (blu della palette), dalla banda vicina alla lontana
BAND = ['#0d3d78', '#2a78d6', '#7fb0ea', '#cfe0f6']


def save(fig, name):  # noqa: F811  anteprime nella cartella del capitolo
    os.makedirs(style.OUT, exist_ok=True)
    fig.savefig(f"{style.OUT}/{name}.pdf")
    preview = os.path.join(HERE, "anteprime")
    os.makedirs(preview, exist_ok=True)
    fig.savefig(f"{preview}/{name}.png", dpi=170)
    plt.close(fig)
