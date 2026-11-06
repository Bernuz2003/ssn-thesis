# Figure del Capitolo 5 (memoria temporale relazionale)

Riusano `style.py` e `load.py` di `../cap4`. Output in `images/cap5/`, anteprime PNG in `anteprime/`.

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap5 && for f in fig_*.py; do $PY $f; done

- `fig_bande.py`: ritardi del mixer a ritardi e bande della memoria a intervalli in tempo fisico
  (ritardi e durata dei bin dalle config risolte dei run VC).
- `fig_disponibilita.py`: storia disponibile per banda, calcolata sulle lunghezze dei campioni di validazione
  in `artifacts/reports/c0_e0_vc/<dataset>/VC/clean_samples.csv` (colonna `steps`).
