"""Riformulazione della self-attention nel dominio spiking (sezione 1.3.3).

Tutte le matrici disegnate sono calcolate davvero, su un piccolo esempio casuale con N=6 token e
D=5 canali, secondo le formule degli articoli:
  VSA   softmax(Q K^T / sqrt(d)) V                     (Vaswani et al. 2017)
  SSA   SN(Q_s K_s^T V_s * beta)                       (Zhou et al. 2023)
  SDSA  SN(SUM_token(Q_s o K_s)) o V_s                 (Yao et al. 2023, eq. 13)
  QKTA  SN(SUM_canali(Q_s)) o K_s                      (Zhou et al. 2024)
La soglia degli strati SN e' scelta per rendere leggibile l'esempio.
"""
from schemi import *  # noqa

N, D = 6, 5
rng = np.random.default_rng(7)
P = 0.5
Qs = (rng.random((N, D)) < P).astype(int)
Ks = (rng.random((N, D)) < P).astype(int)
Vs = (rng.random((N, D)) < P).astype(int)
Qr, Kr, Vr = rng.normal(size=(N, D)), rng.normal(size=(N, D)), rng.normal(size=(N, D))


def norm01(a):
    return (a - a.min()) / (a.max() - a.min())


# VSA
raw = Qr @ Kr.T / np.sqrt(D)
soft = np.exp(raw) / np.exp(raw).sum(1, keepdims=True)
Yr = soft @ Vr
# SSA
QK = Qs @ Ks.T
QKV = QK @ Vs
Yssa = (QKV * 0.5 >= 1.0).astype(int)
# SDSA
QoK = Qs * Ks
sum_tok = QoK.sum(0, keepdims=True)
Ac = (sum_tok >= 2).astype(int)
Ysd = Ac * Vs
# QKTA
sum_can = Qs.sum(1, keepdims=True)
At = (sum_can >= 3).astype(int)
Yqk = At * Ks
print("SDSA somma per canale", sum_tok, "maschera", Ac)
print("QKTA somma per token", sum_can.ravel(), "maschera", At.ravel())
print("valore massimo in QK^T", QK.max(), "in QK^T V", QKV.max())

W, H = WIDTH, 4.3
fig, ax = canvas(W, H)
CW, CH = 0.082, 0.082
XS = [1.24, 2.34, 3.40, 4.28, 5.30]
RY = [3.72, 2.74, 1.76, 0.78]
GAP = 0.12


def riga(y, nome, modello):
    label(ax, 0.0, y + 0.08, nome, fs=9, ha="left")
    label(ax, 0.0, y - 0.09, modello, fs=7.5, color=INK2, ha="left")


def freccia(i, j, y, testo, testo2=None, xfrom=None, xto=None):
    x0 = xfrom if xfrom is not None else XS[i]
    x1 = xto if xto is not None else XS[j]
    arrow(ax, x0, y, x1, y)
    xm = (x0 + x1) / 2
    if testo2:
        label(ax, xm, y + 0.22, testo)
        label(ax, xm, y + 0.09, testo2)
    else:
        label(ax, xm, y + 0.10, testo)


def nota(x, y, testo):
    label(ax, x, y - 0.335, testo, fs=7, color=INK2)


def coppia(y, a, b, kind, la, lb):
    dx = 0.235
    tensor(ax, XS[0] - dx, y, a, kind, CW * 0.8, CH)
    tensor(ax, XS[0] + dx, y, b, kind, CW * 0.8, CH)
    label(ax, XS[0] - dx, y + 0.34, la)
    label(ax, XS[0] + dx, y + 0.34, lb)
    return XS[0] + dx + D * CW * 0.8 / 2 + 0.03


# VSA
y = RY[0]
riga(y, "VSA", "Transformer")
x0 = coppia(y, norm01(Qr), norm01(Kr), "real", r"$Q$", r"$K$")
freccia(0, 1, y, r"$QK^\top$", xfrom=x0, xto=XS[1] - N * CW / 2 - 0.03)
tensor(ax, XS[1], y, norm01(raw), "real", CW, CH)
nota(XS[1], y, r"$N\times N$")
freccia(1, 2, y, "softmax", xfrom=XS[1] + N * CW / 2 + 0.03, xto=XS[2] - N * CW / 2 - 0.03)
tensor(ax, XS[2], y, soft, "real", CW, CH, outline=True)
nota(XS[2], y, r"matrice $N\times N$")
freccia(2, 4, y, r"$\cdot\,V$", xfrom=XS[2] + N * CW / 2 + 0.03, xto=XS[4] - D * CW / 2 - 0.03)
tensor(ax, XS[4], y, norm01(Yr), "real", CW, CH)
nota(XS[4], y, r"uscita")

# SSA
y = RY[1]
riga(y, "SSA", "Spikformer")
x0 = coppia(y, Qs, Ks, "bin", r"$Q_s$", r"$K_s$")
freccia(0, 1, y, r"$Q_sK_s^\top$", xfrom=x0, xto=XS[1] - N * CW / 2 - 0.03)
tensor(ax, XS[1], y, QK, "int", CW, CH, outline=True)
nota(XS[1], y, r"matrice $N\times N$")
freccia(1, 2, y, r"$\cdot\,V_s\,\beta$", xfrom=XS[1] + N * CW / 2 + 0.03, xto=XS[2] - D * CW / 2 - 0.03)
tensor(ax, XS[2], y, QKV, "int", CW, CH)
nota(XS[2], y, r"$N\times D$")
freccia(2, 4, y, "SN", xfrom=XS[2] + D * CW / 2 + 0.03, xto=XS[4] - D * CW / 2 - 0.03)
tensor(ax, XS[4], y, Yssa, "bin", CW, CH)
nota(XS[4], y, r"uscita")

# SDSA
y = RY[2]
riga(y, "SDSA", "Spike-driven T.")
x0 = coppia(y, Qs, Ks, "bin", r"$Q_s$", r"$K_s$")
freccia(0, 1, y, r"$\odot$", xfrom=x0, xto=XS[1] - D * CW / 2 - 0.03)
tensor(ax, XS[1], y, QoK, "bin", CW, CH)
nota(XS[1], y, r"$N\times D$")
freccia(1, 2, y, "somma", "sui token", xfrom=XS[1] + D * CW / 2 + 0.03, xto=XS[2] - D * CW / 2 - 0.03)
tensor(ax, XS[2], y, sum_tok, "int", CW, CH)
nota(XS[2], y, r"conteggi")
freccia(2, 3, y, "SN", xfrom=XS[2] + D * CW / 2 + 0.03, xto=XS[3] - D * CW / 2 - 0.03)
tensor(ax, XS[3], y, Ac, "bin", CW, CH, outline=True)
nota(XS[3], y, r"vettore su $D$ canali")
freccia(3, 4, y, r"$\odot\,V_s$", xfrom=XS[3] + D * CW / 2 + 0.03, xto=XS[4] - D * CW / 2 - 0.03)
tensor(ax, XS[4], y, Ysd, "bin", CW, CH)
nota(XS[4], y, r"uscita")

# QKTA
y = RY[3]
riga(y, "QKTA", "QKFormer")
tensor(ax, XS[0], y, Qs, "bin", CW, CH)
label(ax, XS[0], y + 0.34, r"$Q_s$")
x0 = XS[0] + D * CW / 2 + 0.03
freccia(0, 1, y, "somma", "sui canali", xfrom=x0, xto=XS[1] - CW / 2 - 0.03)
tensor(ax, XS[1], y, sum_can, "int", CW, CH)
nota(XS[1], y, r"conteggi")
freccia(1, 2, y, "SN", xfrom=XS[1] + CW / 2 + 0.03, xto=XS[2] - CW / 2 - 0.03)
tensor(ax, XS[2], y, At, "bin", CW, CH, outline=True)
nota(XS[2], y, r"vettore su $N$ token")
freccia(2, 4, y, r"$\odot\,K_s$", xfrom=XS[2] + CW / 2 + 0.03, xto=XS[4] - D * CW / 2 - 0.03)
tensor(ax, XS[4], y, Yqk, "bin", CW, CH)
nota(XS[4], y, r"uscita")

# legenda
yl = 0.08
xl = 1.28
for col, txt, dx in [(BIN, r"binario", 0.72), (INT, r"a pi{\`u} bit", 0.84), (REAL, r"reale", 0.62)]:
    ax.add_patch(Rectangle((xl, yl - 0.05), 0.10, 0.10, fc=col, ec="none", zorder=3))
    label(ax, xl + 0.15, yl, txt, ha="left", color=INK2)
    xl += dx
ax.add_patch(Rectangle((xl, yl - 0.05), 0.10, 0.10, fc="none", ec=INK, lw=1.4, zorder=3))
label(ax, xl + 0.15, yl, r"struttura che svolge il ruolo di attenzione", ha="left", color=INK2)

save(fig, "cap1_operatori_attenzione")
