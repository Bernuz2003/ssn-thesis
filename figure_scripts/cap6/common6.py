"""Riusa stile del Capitolo 4 e utilita' del Capitolo 5, con output in images/cap6."""
import os, sys
HERE6 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE6, '..', 'cap5'))
from common5 import *  # noqa
import style
style.OUT = os.path.join(HERE6, '..', '..', 'images', 'cap6')

# Colori dei due dataset (slot 1 e 2 della palette validata), come nella figura della compressione
DSCOL = {'dvslip': "#2a78d6", 'thu_eact_50_chl': "#eb6834"}
DSNAME = {'dvslip': 'DVS-Lip', 'thu_eact_50_chl': 'THU-EACT-50-CHL'}


def save(fig, name):  # noqa: F811
    os.makedirs(style.OUT, exist_ok=True)
    fig.savefig(f"{style.OUT}/{name}.pdf")
    preview = os.path.join(HERE6, "anteprime")
    os.makedirs(preview, exist_ok=True)
    fig.savefig(f"{preview}/{name}.png", dpi=170)
    plt.close(fig)
