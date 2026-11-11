"""Tre modalita' di accesso al passato (sezione 1.5). Schema illustrativo, non risultati.

(a) sola memoria neuronale: il peso di ogni istante e' lambda^n, con lambda = 0,5
    (equazione dell'espansione del potenziale, sezione 1.2.2).
(b) attenzione a blocchi: confronto congiunto dentro ogni blocco (STAtten, Lee et al. 2025), qui B = 4.
(c) mixing a ritardi: lettura di istanti scelti, qui con i ritardi 1, 2, 4, 8 del mixer del modello
    di riferimento; lo spessore dei collegamenti indica pesi diversi ed e' illustrativo.
"""
from schemi import *  # noqa

W, H = WIDTH, 3.42
fig, ax = canvas(W, H)
NX = 9                     # istanti da t-8 a t
X0, DX = 1.72, 0.485
XS = [X0 + i * DX for i in range(NX)]
S = 0.26                   # lato dei quadrati
LAM = 0.5
PAST = "#d9d8d4"


def nodo(x, y, fc, ec=None, lw=0.5):
    ax.add_patch(Rectangle((x - S / 2, y - S / 2), S, S, fc=fc, ec=ec or tint(INK2, 0.5), lw=lw, zorder=4))


def catena(y, forte=False):
    """Stato dei neuroni: collega ogni passo al successivo."""
    yc = y - S / 2 - 0.13
    col = INK2 if forte else tint(REAL, 0.55)
    lw = 1.3 if forte else 0.7
    arrow(ax, XS[0] - 0.20, yc, XS[-1] + 0.20, yc, color=col, lw=lw)
    for x in XS:
        arrow(ax, x, y - S / 2, x, yc, color=col, lw=0.5 if not forte else 0.8, head=False)
        ax.add_patch(Circle((x, yc), 0.022 if not forte else 0.03, fc=col, ec="none", zorder=4))
    label(ax, XS[0] - 0.26, yc, "stato", fs=7, color=INK2, ha="right")


def arco(i, j, y, col, lw, head=False):
    L = (j - i) * DX
    h = {1: 0.12, 2: 0.19, 3: 0.25, 4: 0.30, 8: 0.40}[j - i]
    rad = 2 * h / L
    arrow(ax, XS[i], y + S / 2 + 0.005, XS[j], y + S / 2 + 0.005, color=col, lw=lw, head=head,
          conn=f"arc3,rad=-{rad:.3f}", z=3)


def titolo(y, a, b):
    label(ax, 0.0, y + 0.09, a, fs=8.5, ha="left")
    label(ax, 0.0, y - 0.08, b, fs=7, color=INK2, ha="left")


# (a) sola memoria neuronale
y = 2.86
titolo(y, r"(a) memoria neuronale", r"il passato in un solo stato")
catena(y, forte=True)
for i, x in enumerate(XS):
    n = NX - 1 - i
    nodo(x, y, tint(BIN, max(LAM ** n, 0.04)))
for i in (8, 7, 6, 5):
    n = NX - 1 - i
    testo = "1" if n == 0 else rf"$\lambda^{{{n}}}$" if n > 1 else r"$\lambda$"
    label(ax, XS[i], y + S / 2 + 0.10, testo, fs=7, color=INK2)

# (b) attenzione a blocchi
y = 1.74
titolo(y, r"(b) attenzione a blocchi", r"confronto nel blocco")
catena(y)
blocchi = [(1, 4), (5, 8)]
for k, (a, b) in enumerate(blocchi):
    corrente = k == 1
    col = BIN if corrente else tint(BIN, 0.55)
    x0 = XS[a] - S / 2 - 0.05
    w = (b - a) * DX + S + 0.10
    ax.add_patch(FancyBboxPatch((x0, y - S / 2 - 0.05), w, S + 0.10, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="none", ec=col, lw=0.9, zorder=3))
    for i in range(a, b + 1):
        for j in range(i + 1, b + 1):
            arco(i, j, y, col, 0.8)
for i, x in enumerate(XS):
    if i == NX - 1:
        fc = BIN
    elif i >= 5:
        fc = tint(BIN, 0.40)
    elif i >= 1:
        fc = tint(BIN, 0.18)
    else:
        fc = PAST
    nodo(x, y, fc)

# (c) mixing a ritardi
y = 0.62
titolo(y, r"(c) mixing a ritardi", r"lettura di istanti scelti")
catena(y)
RIT = {1: 2.2, 2: 0.9, 4: 1.6, 8: 0.8}
for d, lw in RIT.items():
    arco(NX - 1 - d, NX - 1, y, BIN, lw, head=True)
for i, x in enumerate(XS):
    d = NX - 1 - i
    fc = BIN if d == 0 else tint(BIN, 0.40) if d in RIT else PAST
    nodo(x, y, fc)

# asse del tempo
ya = 0.11
for i, x in enumerate(XS):
    d = NX - 1 - i
    label(ax, x, ya, r"$t$" if d == 0 else rf"$t{{-}}{d}$", fs=7.5, color=INK2)

save(fig, "cap1_accesso_passato")
