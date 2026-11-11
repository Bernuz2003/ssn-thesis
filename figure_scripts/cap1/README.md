# Figure del Capitolo 1, sezioni 1.1, 1.2, 1.3 e 1.5

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
- `schemi.py`: primitive comuni agli schemi a blocchi della sezione 1.3 (coordinate in pollici).
- `fig_attenzione_dizionario.py`: dizionario e attenzione a confronto, con valori illustrativi.
- `fig_operatori_attenzione.py`: VSA, SSA, SDSA e QKTA calcolate davvero su un esempio casuale (seed 7, N=6, D=5),
  secondo le formule degli articoli. La soglia degli strati SN e' scelta per leggibilita'.
- `fig_residui.py`: residui SEW e MS secondo le equazioni 4 e 5 di Spikingformer.
- `fig_gerarchia.py`: Spikformer e QKFormer secondo le dimensioni riportate negli articoli originali.
- `fig_accesso_passato.py`: memoria neuronale, attenzione a blocchi e mixing a ritardi a confronto, schema illustrativo.
