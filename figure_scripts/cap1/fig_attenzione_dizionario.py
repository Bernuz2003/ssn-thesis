"""L'attenzione come ricerca graduale in un dizionario (sezione 1.3.1). Valori illustrativi."""
from schemi import *  # noqa

W, H = WIDTH, 3.0
fig, ax = canvas(W, H)

XQ, XK, XP, XV, XR = 1.32, 2.42, 3.17, 3.92, 5.12
BW, BH = 0.70, 0.27
G = lambda s: r"{\guillemotleft}" + s + r"{\guillemotright}"

for x, t in [(XQ, "richiesta"), (XK, "chiavi"), (XP, "pesi"), (XV, "valori"), (XR, "risultato")]:
    label(ax, x, H - 0.11, t, fs=8.5, color=INK2)


def gruppo(y0, titolo, sotto, q, keys, vals, pesi, risultato, risultato_w):
    ys = [y0 + 0.37, y0, y0 - 0.37]
    label(ax, 0.0, y0 + 0.10, titolo, fs=8.5, ha="left")
    label(ax, 0.0, y0 - 0.08, sotto, fs=7.5, color=INK2, ha="left")
    box(ax, XQ, y0, BW, BH, q, fc=tint(BIN, 0.12), ec=BIN)
    box(ax, XR, y0, risultato_w, BH + 0.05, risultato, fc=tint(BIN, 0.12), ec=BIN, fs=7.5)
    for y, k, v, p in zip(ys, keys, vals, pesi):
        attivo = p > 0
        col = BIN if attivo else MUTED
        lw = 0.5 + 3.0 * p if attivo else 0.6
        ls = "-" if attivo else (0, (2, 2))
        arrow(ax, XQ + BW / 2, y0, XK - BW / 2, y, color=col if attivo else GRID, lw=0.8, ls=ls, head=False)
        box(ax, XK, y, BW, BH, k)
        box(ax, XV, y, BW, BH, v)
        arrow(ax, XK + BW / 2, y, XV - BW / 2, y, color=col, lw=lw, ls=ls, head=False)
        arrow(ax, XV + BW / 2, y, XR - risultato_w / 2, y0, color=col, lw=lw, ls=ls, head=False)
        testo = num(p, 1) if p not in (0, 1) else f"{p:d}"
        ax.text(XP, y, rf"${testo}$", fontsize=7.5, ha="center", va="center", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"))


gruppo(2.25, r"(a) dizionario", r"esito s{\`i} o no", G("gatto"),
       [G("cane"), G("gatto"), G("pesce")], [G("dog"), G("cat"), G("fish")], [0, 1, 0], G("cat"), 0.70)
gruppo(0.75, r"(b) attenzione", r"esito graduale", r"Query $q$",
       [r"Key $k_1$", r"Key $k_2$", r"Key $k_3$"], [r"Value $v_1$", r"Value $v_2$", r"Value $v_3$"],
       [0.7, 0.2, 0.1], r"$0{,}7\,v_1{+}0{,}2\,v_2{+}0{,}1\,v_3$", 1.30)

save(fig, "cap1_attenzione_dizionario")
