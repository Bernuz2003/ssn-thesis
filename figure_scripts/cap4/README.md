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
- `data/DvsGesture/events/train/6/user17_led__06.npz`: campione raw per le figure dei tre clock e dell'aliasing.
- Valori di Macro-F1 in `fig_risultati.py`: summary.json dei run elencati nel commento in testa al capitolo.

Le anteprime PNG finiscono in `anteprime/` (non incluse nella tesi).
