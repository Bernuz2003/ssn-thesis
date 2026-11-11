"""Stile comune (da ../cap4) con output in images/cap2 e anteprime locali."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cap4'))
import style
style.OUT = os.path.join(HERE, '..', '..', 'images', 'cap2')
from style import *  # noqa

RESEARCH = '/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research'
ON = '#e34948'    # luce aumentata
OFF = '#2a78d6'   # luce diminuita
MID = '#f0efec'


def save(fig, name):  # noqa: F811
    os.makedirs(style.OUT, exist_ok=True)
    fig.savefig(f"{style.OUT}/{name}.pdf")
    preview = os.path.join(HERE, "anteprime")
    os.makedirs(preview, exist_ok=True)
    fig.savefig(f"{preview}/{name}.png", dpi=170)
    plt.close(fig)
