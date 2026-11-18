# Figure del Capitolo 3

Riusano `style.py` di `../cap4`; i dati sono letti dal repository di ricerca (`common3.py`).
Output in `images/cap3/`, anteprime PNG in `anteprime/`.

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap3 && for f in fig_*.py; do $PY $f; done

- `fig_diagnosi.py`: quote di MAC e accumuli potenziali per blocco nella baseline (profilo comune) e Macro-F1
  delle varianti diagnostiche su DVS-Lip.
- `fig_modello_riferimento.py`: schema del modello di riferimento, frontend piramidale e mixer a ritardi,
  verificato su `src/etsr/models/{layers,temporal}.py`.
- `fig_percorso.py`: Macro-F1 e costi relativi lungo la catena baseline, frontend piramidale, mixer a ritardi,
  convoluzione depthwise (profili in `artifacts/reports/architecture_evolution/`). Le MAC del profilo
  comprendono già quelle del mixer.
- `fig_solidita.py`: baseline e modello di riferimento sui quattro dataset (barre = stesso seed per i due modelli,
  punti = ripetizioni ulteriori) e Macro-F1 in funzione dei parametri su DVS-Lip per 128, 192 e 256 canali.
