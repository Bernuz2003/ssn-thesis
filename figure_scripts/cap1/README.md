# Figure del Capitolo 1, sezioni 1.1 (event camera e rappresentazioni) e 1.2 (dinamica neuronale)

Riusano `style.py` di `../cap4`. Output in `images/cap1/`, anteprime PNG in `anteprime/`.

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap1 && for f in fig_*.py; do $PY $f; done

- `fig_generazione_eventi.py`: modello ideale del pixel DVS (soglia logaritmica C=0,2) su un segnale sintetico.
- `fig_partizioni.py`: quattro modi di suddividere un flusso sintetico (seed 3), puramente illustrativo.
- `fig_rappresentazioni.py`: event frame, voxel grid (B=5) e time surface (tau=50 ms) sull'intervallo
  2,0-2,1 s del campione reale `data/DvsGesture/events/train/6/user17_led__06.npz` del repository di ricerca.
- `fig_dinamica_lif.py`: neurone IF e LIF (lambda=0,5, eta=1, soglia 1, soft reset) con tre ingressi da 0,6
  consecutivi o distanziati. Stampa i potenziali usati nel testo della sezione 1.2.1.
- `fig_surrogata.py`: gradino e sigmoide nel passaggio in avanti, derivata surrogata sigmoidale per
  alpha = 2, 4, 8 nel passaggio all'indietro (alpha=4 e' il valore della baseline, capitolo 2).
