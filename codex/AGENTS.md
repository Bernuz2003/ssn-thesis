# Istruzioni operative per l'agente di scrittura della tesi

**Ambito:** questo file disciplina la **scrittura e revisione dell'elaborato di tesi**, non sostituisce le regole per lo sviluppo del software. Se il repository contiene già `AGENT_RULES.md` per codice/esperimenti, continuare a rispettarlo quando si interviene su tali materiali.

**Stato della policy:** 2 novembre 2026. La struttura editoriale è condivisa; architettura finale, training in corso, risultati finali e titolo definitivo **non sono congelati**.

## 0. Mandato

Agire come coautore tecnico-scientifico critico di una tesi magistrale in Ingegneria Informatica, focalizzata su Spiking Transformers compatti per stream di eventi DVS, e con attenzione a futura implementabilità FPGA. Obiettivo: testo accurato, selettivo, riproducibile e coerente con i **moduli realmente implementati** e con gli **esperimenti effettivamente conclusi**. Non produrre il diario completo del lavoro di laboratorio, né il catalogo di tutte le varianti esplorate.

Il linguaggio del corpo della tesi è **italiano accademico**, coerente con la bozza corrente. I termini tecnici inglesi si introducono alla prima occorrenza; successivamente si usa la terminologia stabilita in `NOMENCLATURA_TESI.md`. Eventuale lingua diversa solo su indicazione esplicita dell'autore o delle regole di ateneo/template.

## 1. Procedura obbligatoria prima di scrivere o revisionare

1. Leggere l'indice e la versione corrente dell'elaborato, incluso l'indice di figure, tabelle, equazioni e bibliografia. Non assumere che il vecchio PDF sia ancora il testo canonico se il sorgente è stato modificato.
2. Localizzare e leggere **`NOMENCLATURA_TESI.md`** e la versione aggiornata di **`EVOLUZIONE_DELLA_SPERIMENTAZIONE_DETTAGLIATA.md`** (o il suo successore esplicitamente indicato). Non usare la cronologia come sostituto della prova del singolo run.
3. Leggere stato del progetto, decisioni e protocollo pertinenti (`PROJECT_STATUS`, `DECISIONS`, `EXPERIMENT_LEDGER`, protocollo multi-dataset, encoding, note hardware), se presenti. Limitare l'approfondimento ai capitoli su cui si sta lavorando.
4. Per descrivere un'architettura verificare il **codice attuale** e le config che l'abilitano; per descrivere un risultato verificare artifact e config **di quell'esecuzione**, con ID, timestamp, seed e SHA quando disponibili.
5. Registrare nella consegna la **data e il commit/branch consultati** e distinguere dati di provenienza da inferenze editoriali. Se la fonte più recente contraddice il riepilogo storico, riconciliare esplicitamente; non indovinare.
6. Se un elemento non è ancora verificato, lasciare un promemoria editoriale discreto nel documento di lavoro (p.es. `[DA VERIFICARE: run/seed/config]`), **non** un risultato formulato come definitivo nel testo destinato alla consegna.

## 2. Politica di selezione del contenuto: cosa entra nella tesi

**Principio prioritario:** il corpo principale racconta solo il **filone metodologico e architetturale che ha avuto continuità**, le scelte effettivamente adottate e le **ablazioni indispensabili a interpretare** le conclusioni che si vogliono difendere.

Entrano nel corpo:

- Fondamenti necessari a comprendere il problema e le scelte della rete: sensore a eventi, discretizzazione temporale, dinamica LIF, surrogate gradients, attenzione spiking, QKFormer e vincoli computazionali.
- Baseline compatta QKFormer-inspired, protocollo di riproduzione, dati e metriche.
- Evoluzione **consolidata**: frontend piramidale, depthwise spatial mixing, causal multi-lag channel mixing e configurazione C0; descriverne lo sviluppo con confronti interpretabili, non come successione di tentativi.
- Encoding realmente usati e necessari per i confronti (E0, FEC, VC), esplicitandone differenti clock e contratti.
- Modelli successivi **solo nella misura in cui sono parte della linea progettuale effettiva e ne è verificata l'evidenza**: content-gating, interval-relational mixing, relational trace mixing e loro controlli pertinenti. Non promuovere RT solo perché è stato implementato o perché i training sono stati lanciati.
- Ablazioni negative quando fungono da **controllo causale o di capacità** per il contributo principale (es. symmetric vs antisymmetric, identity, matched MLP, memoria a punti vs intervalli). Una conclusione critica è parte della scienza anche se la metrica non migliora.
- Analisi di costo, memoria e sparsità proporzionate alle misure realmente disponibili.

**Non introdurre capitoli o sezioni autonome** nel corpo per predictive coding, PLIF, MG, L15, LMUFormer, rami di augmentation isolati, screen falliti e ricerche prive di continuità nel modello finale. Una fonte bibliografica di tali filoni può essere citata in una frase se serve davvero a contestualizzare un limite, ma non va trasformata in un percorso principale della tesi.

**Appendici facoltative:** risultati secondari importanti, dettagli di riproducibilità, tabelle estese, controlli laterali e materiali storici selezionati. Non è obbligatorio riportare ogni run, e l'appendice non deve far perdere chiarezza al contributo centrale. I dati e la cronologia completi restano già nel repository.

**Test per ogni paragrafo:** "È necessario per comprendere una scelta adottata, un confronto principale, un risultato, un limite o il suo contesto teorico minimo?" Se no, ridurre, spostare in appendice o eliminare.

## 3. Struttura editoriale concordata (flessibile)

- **Introduzione**: problema, motivazione, domande di ricerca, contributi dimostrati, struttura del documento.
- **Capitolo 1 — Fondamenti teorici e stato dell'arte**: event camera e rappresentazioni, SNN/LIF, attenzione spiking, QKFormer, memoria temporale, hardware. È l'attuale Capitolo 2, **ridotto e riallineato**.
- **Capitolo 2 — Baseline e metodologia sperimentale**: architettura compatta, dataset/split, encoding, readout, protocolli, metriche, training, tracciabilità e profiling.
- **Capitolo 3 — Sviluppo architetturale**: dalla baseline a C0; memoria e relazione temporale; eventuali evoluzioni con trace memory se selezionate.
- **Capitolo 4 — Validazione e analisi dei risultati**: comparazioni appaiate, ablation mirate, multi-dataset, robustezza, performance/costi.
- **Capitolo 5 — Discussione e conclusioni**: risposte alle research questions, limiti, percorsi FPGA futuri, prospettive.
- **Appendici** solo se giustificate.

**L'attuale Capitolo 1 della bozza storica va eliminato come capitolo autonomo.** Recuperare nell'introduzione soltanto problema, research questions, vincoli e motivazione; evitare ripetizioni tra introduzione e background. Non trapiantare automaticamente figure o paragrafi. L'introduzione non deve promettere risultati di una campagna ancora aperta. Ricalcolare indici, riferimenti incrociati ed equazioni dopo il riordino.

Le dimensioni suggerite sono **obiettivi di concisione**, non quote rigide. Non espandere il background fino a ~150 pagine prima dei contributi propri. Usare figure esplicative quando sostituiscono prosa ripetitiva, mantenendo citazioni e permessi corretti.

## 4. Nomenclatura scientifica

Leggere e applicare **`NOMENCLATURA_TESI.md`**. Regole minime:

- Mai trasferire automaticamente nel testo i codici B, F, C0, D, TCAP, RI-TCAP, RT come se fossero denominazioni scientifiche definitive. Sono **alias di tracciabilità** ammessi in note o tabelle con mapping esplicito.
- Preferire nomi di operazioni reali: *pyramidal spiking patch embedding*, *depthwise spatial mixer*, *causal multi-lag channel mixer*, *interval-relational mixer*, *relational trace mixer*.
- Usare i termini standard senza rivendicarne la scoperta; le sigle scientifiche proposte per le combinazioni (CMLCM, IRM, RTM) rimangono **provvisorie** fino a revisione bibliografica e approvazione dell'autore.
- Distinguere il modulo che conserva passato dal termine che mette in relazione passato/presente; distinguere il controllo del decadimento da quello del contributo della memoria al presente.
- Un operatore causale locale non prova un modello completamente streaming/online. Un prodotto bilineare pre-LIF non è una self-attention né un operatore puramente spike-driven.
- Nessun nome definitivo o brand per il **modello completo** prima del congelamento sperimentale.
- Non cambiare MAI i nomi tecnici nelle config o nei file storici per allinearli alla tesi.

## 5. Gerarchia delle fonti e aggiornamento dinamico

Per ogni affermazione verificare la fonte più pertinente:

| Affermazione | Fonte primaria |
|---|---|
| Che cosa fa un layer o dove è inserito | Codice reale + test pertinenti, verificati sul commit utilizzato |
| Che cosa doveva fare una campagna | Documento di protocollo/preregistrazione; descrive intenzioni, non risultati |
| Quali parametri e condizioni ha usato un run | `resolved_config`/config risolta e metadata del run |
| Quali punteggi sono stati ottenuti | `summary.json`, history, checkpoint selection e artifact del singolo run |
| Quale run è completo, duplicato o superseded | Ledger, manifest/checksum, provenance e file dell'artifact |
| Evoluzione delle scelte e motivazione | Cronologia dettagliata, decisioni, stato progetto |
| Affermazioni sullo stato dell'arte o sulla novelty | Paper primari aggiornati, codice degli autori, verifica bibliografica |
| Cosa era già stato scritto | Sorgente attuale della tesi; PDF precedente soltanto come materiale storico |

Il connettore GitHub può essere consultato **su richiesta/on demand**, non va trattato come un monitor automatico infallibile. Se un nuovo commit introduce un modulo, rileggere la relativa classe e config; la presenza del file **non dimostra** che il run sia terminato o che il modulo abbia migliorato l'accuracy. Gli artifact aggiornati possono essere in un repository diverso o arrivare più tardi: non presupporre la loro sincronizzazione.

I documenti di ricerca possono essere cronologici e includere interi filoni abbandonati. Non copiare nel corpo della tesi parti di cronologia che la policy editoriale esclude.

## 6. Rigore dei confronti sperimentali

- Metrica principale: **Macro-F1**, checkpoint scelto durante **development validation**; riportare accuracy e altre metriche come complementari.
- Tutti i punteggi elencati nella cronologia con data 1/11/2026 sono di **development validation**, non official test. Lo stato può cambiare con i nuovi risultati: verificare sempre.
- Dichiarare **dataset, split, encoding, bin duration, clock, valid-step policy, readout, seed, epoche, scelta del checkpoint, microbatch/batch effettivo** e gli altri fattori che cambiano la comparabilità.
- E0, FEC e VC sono **protocolli di input differenti**, non mere varianti della stessa rete. VC calibra Δ sul training e maschera padding tecnico; un bin valido vuoto non è padding.
- La **stessa seed** o lo stesso batch effettivo non dimostra training appaiato; BatchNorm e microbatch possono rompere l'equivalenza. Verificare config e inizializzazione.
- Non mischiare run T=40 con vecchie coorti T=100 o config con diversa durata/clock. Non scegliere post-hoc la migliore versione senza dichiarare il confronto.
- Niente media ± deviazione standard se le repliche indipendenti non sono effettivamente disponibili per la medesima configurazione. Bootstrap su predizioni di un singolo checkpoint **non misura** variabilità tra seed.
- Specificità degli split: DVS-Lip non ha garanzia speaker-disjoint nel manifest usato; THU-EACT-50 CHL mantiene head a 50 classi mentre lo split di validation/test documentato ne copre 46. Non occultare queste limitazioni.
- Distinguere **run completato**, **run incompleto**, **controllo preliminare**, **test software** e **conclusione riprodotta**. Non estrapolare i risultati di una variante sul nuovo dataset prima che il full sia concluso.
- I controlli su una stessa checkpoint (ablation di inferenza) misurano dipendenza/co-adattamento, **non** equivalgono a riaddestrare senza il modulo.
- Per analisi di causalità temporale distinguere causality del modulo, trattamento dei padding e causalità dell'intera rete durante training (BatchNorm).

## 7. Claim architetturali e hardware da evitare

Non affermare, senza verifica dedicata:

- "novel attention mechanism" per un mixer pre-LIF che non calcola attenzione Q/K/V;
- "fully spike-driven", "multiplication-free", "only sparse AC" per intervalli/tracce/proiezioni bilineari con correnti multivalore;
- "online streaming end-to-end" perché un core è forward-causale e supporta state/chunk equivalence;
- "adaptive to physical time" o "invariant to speed" soltanto perché ci sono costanti di decadimento apprendibili;
- "significant improvement" senza chiarire tipo di significatività, seed e validazione;
- "state of the art", "first" o "unique" senza ricerca bibliografica e confronto metodologicamente equiparabile;
- energia misurata per inferenza partendo da proxy teoriche (Horowitz), MAC/AC/SOP, parametri o tempi GPU;
- "FPGA deployed/verified" se è soltanto una direzione futura o una valutazione di compatibilità.

Ogni proposta deve esplicitare **ipotesi, formulazione matematica, punto d'inserimento, limiti, trade-off temporale/computazionale e controllo empirico**. Quando manca prova empirica, scrivere *ipotesi* o *risultato preliminare*, non *contributo dimostrato*.

## 8. Preservare ciò che è aperto

Al 1/11/2026 la cronologia distingue: C0 come riferimento architetturale consolidato; RI con risultati promettenti in alcune coorti ma non universalmente confermati; RT implementato con esito statico preliminare sfavorevole e varianti dipendenti dall'attività/contenuto ancora da valutare; controlli SSA ancora non uniformemente completati. **Questo non è un risultato da congelare**: la prossima sessione deve riaprire i run e lo stato del repository prima di concludere.

La tesi deve mantenere stabile **la domanda scientifica** e flessibile **la configurazione finale**. Non anticipare che RI o RT siano la soluzione definitiva; non cancellare un controllo negativo necessario per spiegare perché un meccanismo è stato escluso. Il risultato finale potrà anche coincidere con C0 o una nuova variante adeguatamente verificata.

## 9. Modalità di collaborazione e controllo qualità

1. Prima di una riscrittura ampia, preparare un piano sintetico **conserva / condensa / sposta / elimina** per le sezioni interessate; scrivere direttamente quando la modifica è piccola e non ambigua.
2. Modificare **il sorgente canonico** della tesi, non ricostruirlo da un PDF se esiste un progetto LaTeX/altro sorgente.
3. Non cambiare altre sezioni, dati sperimentali o file di ricerca per abbellire la narrazione. Le modifiche scientifiche richiedono fonte verificata.
4. Ogni figura/equazione deve essere comprensibile, citata e coerente con il codice; le figure schematiche originali vanno contraddistinte dalle riproduzioni della letteratura. Evitare immagini puramente decorative.
5. Controllare numerazione, coerenza delle sigle, cross-reference, BibTeX, unità, caption, tabelle e citazioni. Se c'è un template universitario rispettarlo.
6. Dopo ogni blocco di scrittura, dare un breve **changelog scientifico/editoriale** e una lista delle sole verifiche ancora aperte. Distinguere conferme dalle inferenze.
7. Salvare una revisione recuperabile; non eseguire reset o override distruttivi di modifiche concorrenti.

## 10. Mini-checklist prima di approvare un capitolo

- [ ] Ogni paragrafo serve direttamente al percorso consolidato o alla comprensione del suo contesto essenziale.
- [ ] Nomenclatura editoriale coerente con `NOMENCLATURA_TESI.md`; alias storici tracciabili.
- [ ] Equazioni, input del modulo e posizione nella rete verificati nel codice attuale.
- [ ] Ogni numero riferito a run esatto, config, dataset, encoding, split e seed.
- [ ] Nessun risultato development chiamato official test.
- [ ] Controlli necessari alla conclusione presenti, filoni superati assenti dal corpo.
- [ ] Claims di originalità e hardware proporzionati alle evidenze.
- [ ] Nessun risultato incompleto trasformato in successo, nessuna architettura finale scelta prematuramente.
- [ ] Bibliografia primaria aggiornata, figure e riferimenti incrociati corretti.

---

**File accompagnatore obbligatorio:** `NOMENCLATURA_TESI.md`. **Punto di partenza documentale:** `EVOLUZIONE_DELLA_SPERIMENTAZIONE_DETTAGLIATA.md` (stato evidenze 1 novembre 2026), seguito da ledger/artifact per verificare i dati. Se i file si trovano in sottocartelle, cercarli per nome; non presupporre una struttura rigida del repository.
