# Figure del Capitolo 2

Riusano `style.py` di `../cap4`. Output in `images/cap2/`, anteprime PNG in `anteprime/`.

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap2 && for f in fig_*.py; do $PY $f; done

- `fig_dataset.py`: un campione di sviluppo per dataset, scelto con durata vicina alla mediana; tre bin della
  finestra fissa (differenza ON - OFF) e frequenza degli eventi sull'intera finestra.
  DVS-Lip (`train/warning/128.npy`) e DVS128 Gesture (`train/1/user07_fluorescent__01.npz`) dal repository
  di ricerca. CIFAR10-DVS (`airplane/cifar10_airplane_318.aedat`) e THU-EACT-50-CHL
  (`A39P17C2-2021_11_06_15_09_46.npy`), entrambi in partizione di addestramento, sono copiati da
  `data/` del checkout SMILIES nella cartella `dati/`, esclusa da git.
- `fig_architettura.py`: (a) blocchi della baseline con dimensioni di uscita, (b) dettaglio del primo patch
  embedding, (c) ripartizione di parametri e stato tra i blocchi su DVS-Lip (valori della tabella 2.3).
  Struttura verificata su `src/etsr/models/{mini_qkformer,layers}.py`; colori dalla palette validata
  (slot 4, 1, 3, 2 in ordine adiacente, testa in grigio).
