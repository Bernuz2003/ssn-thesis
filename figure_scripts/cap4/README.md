# Figure del Capitolo 4 (clock dell'input)

Script che generano le figure in `images/cap4/` a partire dagli artifact del repository di ricerca
`temporal-event-spiking-research` (percorsi assoluti in `load.py` e negli script).
Richiedono il venv del repository di ricerca (numpy, matplotlib) e una distribuzione LaTeX
(le etichette usano `text.usetex` con lmodern).

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap4 && for f in fig_*.py; do $PY $f; done

Fonti dei dati:
- `artifacts/reports/c0_e0_vc/<dataset>/{E0,VC}/clean*`: valutazione FP32 appaiata dei checkpoint C0 seed 42
  (coda vuota, intervento senza coda, quartili di durata, rotazioni E0/VC).
- `artifacts/reports/dvsgesture/mechanistic/dvsgesture_c0_fixed_event_count__*/record.npz`: predizioni FEC.
- `data/DvsGesture/events/train/6/user17_led__06.npz`: campione raw per la figura dell'aliasing.
- `fig_tre_clock.py` genera da sé una registrazione sintetica di 15 s (Poisson non omogeneo, seme fisso) e la
  suddivide con i parametri di DVS128 Gesture; non legge dati.
- Valori di Macro-F1 in `fig_risultati.py`: summary.json dei run elencati nel commento in testa al capitolo.
- `fig_coda.py` produce un'unica figura a tre pannelli (`cap4_coda`): coda vuota, calcolo speso nella coda e
  accuratezza con e senza coda (sostituisce le vecchie `cap4_coda_vuota` e `cap4_dipendenza_coda`).
- `fig_durata_normalizzata.py`: accuratezza per quartili di durata su DVS-Lip e THU-EACT-50-CHL con finestra fissa,
  durata normalizzata (run `FBC_C0_M--vc-horizon`, predizioni in `validation_predictions.csv`) e lunghezza variabile.

Le anteprime PNG finiscono in `anteprime/` (non incluse nella tesi).
