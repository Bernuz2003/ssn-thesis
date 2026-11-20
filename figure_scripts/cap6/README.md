
Aggiornamento 2026-11-19 (capitolo 6 riscritto):
- `finali6.py`: metriche sul test ufficiale dai logit di `artifacts/reports/b_minus_final_inference` (PARAMS, REPEATS
  per le medie degli addestramenti ripetuti). Comprende `lip_kd4_2m_s44` e `lip_c0_2m_kd4`, valutati il 19/11 sui
  SMILIES con `configs/evaluation/b_minus_final_inference_20261119.yaml` (stesso script della valutazione finale).
- `fig_distillazione.py`: `cap6_distillazione` (sezione 6.1), adattamento contro validazione per IRM, auto-ensemble e
  distillazione, e concordanza fra tre addestramenti dell'IRM 2 M con e senza distillazione.
- `fig_capacita.py`: `cap6_capacita` (sezione 6.2), accuratezza sul test in funzione dei parametri.
- `fig_generalizzazione.py` non e' piu' usata: il suo contenuto e' passato alla sezione 5.5 (`cap5/fig_colli_bottiglia.py`).
- `compressione6.py`: rango scelto, risparmi e test denso/compresso di ogni modello (sezione 6.3) dai report della
  valutazione finale; riusato da `../appendice/tabelle_appendice.py` per le tabelle B.2 e B.3.
- `fig_compressione.py`: `cap6_compressione`, perdita in validazione al variare del rango e MAC del sito per mixer a
  ritardi, IRM denso e IRM compresso.
- `fig_stato_arte.py`: `cap6_stato_arte` (sezione 6.5), accuratezza sul test di THU-EACT-50-CHL in funzione dei
  parametri, con i metodi pubblicati verificati sulle versioni arXiv (SMV-EAR, HoloEv-Net, Norrby et al.).
- `fig_errori_lip.py`: `cap6_confusione_lip` (matrice degli errori dell'IRM 2 M distillato sul test, parole ordinate per
  coppie simili, e scomposizione degli errori lungo il percorso dei modelli) e `cap6_durata` (accuratezza per quartile
  del numero di passi, validazione e test). Sezione 6.6.
- `fig_durata.py` e' superata: ora salva in `cap6_durata_vecchia` per non sovrascrivere la figura nuova.
