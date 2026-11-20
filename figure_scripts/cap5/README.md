# Figure del Capitolo 5 (memoria temporale relazionale)

Riusano `style.py` e `load.py` di `../cap4`. Output in `images/cap5/`, anteprime PNG in `anteprime/`.

    PY=/home/emanuele/Scrivania/Tesi/temporal-event-spiking-research/.venv/bin/python
    cd figure_scripts/cap5 && for f in fig_*.py; do $PY $f; done

- `fig_bande.py`: ritardi del mixer a ritardi e bande della memoria a intervalli in tempo fisico
  (ritardi e durata dei bin dalle config risolte dei run VC).
- `fig_disponibilita.py`: storia disponibile per banda, calcolata sulle lunghezze dei campioni di validazione
  in `artifacts/reports/c0_e0_vc/<dataset>/VC/clean_samples.csv` (colonna `steps`).

Aggiornamento 2026-11-15: il capitolo presenta soltanto DVS-Lip e THU-EACT-50-CHL (`common5.DS` filtra i dataset).
- `fig_richiamo_confronto.py`: figura 5.1, schemi del mixer a ritardi, del correlatore di Hassenstein-Reichardt e dell'IRM.
- `fig_relazione.py`: figura della sezione 5.3, lettura geometrica e risposta delle forme antisimmetrica e simmetrica
  su un esempio sintetico (nessun dato sperimentale).
- `risultati5.py`: run della sezione 5.4 (su DVS-Lip l'IRM antisimmetrico di riferimento e' il run 20261112_082513,
  Macro-F1 57,47, addestrato nelle stesse condizioni del modello di riferimento e dei controlli), Macro-F1,
  media delle ultime 16 epoche e confronti appaiati (McNemar esatto). Eseguito da solo stampa tutti i confronti.
- `fig_risultati5.py`: differenza di Macro-F1 rispetto al riferimento per ogni variante, DVS-Lip e THU-EACT-50-CHL.
- `fig_forme_passato.py`: figura della sezione 5.5, peso dei passi passati per bande, tracce esponenziali (orizzonti
  appresi su DVS-Lip, ricavati dai checkpoint delle tracce statiche) e consolidamento (configurazioni illustrative).
- `risultati5.py` comprende anche i run della sezione sulle alternative (tracce, consolidamento, memoria remota, SSA
  temporale, primo stadio, bande in tempo di attivita', orologio appreso, durata normalizzata).

Aggiornamento 2026-11-17 (sezioni 5.5 e 5.7):
- `fig_uso_memoria.py`: `cap5_bande_usate` (variazione di accuratezza azzerando W_k e U_k di una banda nel solo secondo
  sito, validazione, da `artifacts/reports/rt_fbc_comparison/ablations.json`) e `cap5_ordine` (accuratezza sul test
  ufficiale con passi originali, invertiti, rimescolati, da `artifacts/reports/c0_ri_diagnostics/csv/E2.csv`).
- `fig_compressione.py`: `cap5_compressione`, energia dei pesi conservata e differenza di Macro-F1 rispetto al denso al
  variare del rango della base condivisa (`artifacts/reports/b_minus_compression/rank_sweep.json`).
- Su DVS-Lip queste tre fonti usano per ora il checkpoint IRM 20261026_112001: da aggiornare con le valutazioni sul
  checkpoint di riferimento 20261112_082513 (le figure mostrano solo differenze, quindi basta rigenerarle).

Aggiornamento 2026-11-19 (capitolo riscritto: 5.3 risultato, 5.4 controlli, 5.5 colli di bottiglia):
- `fig_uso_memoria.py`: ora produce una sola figura, `cap5_uso_memoria` (2x2: bande spente in validazione, ordine dei
  passi sul test). Su DVS-Lip legge le valutazioni sul checkpoint di riferimento 20261112_082513, eseguite il 19/11 sui
  server SMILIES: `artifacts/reports/lip_ref_bands` (validazione) e `artifacts/reports/lip_ref_diagnostics` (test).
  Le vecchie `cap5_bande_usate` e `cap5_ordine` non sono piu' usate.
- `fig_controlli.py`: statistiche dei dieci controlli della sezione 5.4 (differenza di Macro-F1 con IC bootstrap
  appaiato, p di McNemar, accuratezza di training) salvate in `controlli5.json`, e forest plot `cap5_controlli`.
  Va eseguito prima di `fig_colli_bottiglia.py`.
- `fig_colli_bottiglia.py`: `cap5_colli_bottiglia`, accuratezza di training contro validazione per i controlli e
  quota di clip riconosciute da 0-5 varianti del voto.
- `fig_forme_passato.py`: ridotta a bande e tracce (la memoria di consolidamento non e' piu' nel testo principale).
- `fig_risultati5.py` (`cap5_risultati`) e `fig_compressione.py` non sono usate dal capitolo 5; la compressione
  passa al capitolo 6.
