# Nomenclatura scientifica della tesi

**Stato:** proposta editoriale v1, 2 novembre 2026. **Non è una dichiarazione di novelty né una rinomina del codice.**

## 1. Scopo e principi

Questa tabella traduce le denominazioni operative di sviluppo in termini scientificamente descrittivi da usare nella tesi. Gli identificatori B, F, C0, D, RI, RT, A, B−, B+ e i nomi dei file di configurazione **rimangono invariati** in repository, artifact, checkpoint, script e registri di provenienza. I nuovi termini appartengono esclusivamente alla narrazione scientifica, ai diagrammi e, se approvati, alla loro notazione matematica.

1. Preferire il **termine standard** quando l'operazione esiste già nella letteratura o nella matematica: LIF, spiking self-attention (SSA), depthwise convolution, exponential moving average, low-rank bilinear interaction, antisymmetric bilinear form, causal temporal filtering. Non attribuire tali concetti agli autori della tesi.
2. Dove l'implementazione combina più operazioni, usare un nome **compositivo e verificabile**, evitando termini suggestivi ("predictive", "attention", "fully event-driven", "online") se non sono proprietà dimostrate.
3. Introdurre pochi acronimi; l'acronimo non dà evidenza di novità. Per un modulo discusso una sola volta è preferibile il nome esteso.
4. Non dichiarare la **priorità bibliografica** perché una ricerca non trova un match testuale esatto. Prima di usare espressioni come "we introduce a novel ..." verificare il prior art per funzione, non solo per nome.
5. Separare sempre **famiglia di moduli**, **variante funzionale**, **configurazione architetturale** ed **encoding**. Il nome del modello completo verrà deciso dopo i risultati finali.
6. Nella prima citazione di un modulo introdurre nome inglese, definizione matematica e ruolo; nel resto del testo mantenere la stessa grafia.

## 2. Mappatura: nomi di sviluppo → nomi nella tesi

| Nome nel codice e negli artifact | Terminologia per la tesi (consigliata) | Sigla nel testo | Stato e delimitazione |
|---|---|---|---|
| `B`, Mini-QKFormer | **Compact QKFormer-inspired spiking Transformer baseline** | nessuna nuova | Termine descrittivo. Non è una riproduzione bit-exact di QKFormer. |
| `F`, `frontend: pyramidal` | **Pyramidal spiking patch embedding** / **pyramidal spiking stem** | preferibilmente senza sigla | Operatore architetturale descrittivo, non una nuova categoria di layer. |
| `stage1_mixer: depthwise_conv`, `DWC3` | **Depthwise spatial mixer** (3×3 depthwise convolution) | DWC | Terminologia già standard. |
| `TCAP`, `C0` componente temporale | **Causal Multi-Lag Channel Mixer** | **CMLCM** | Nome proposto per un modulo con somme di proiezioni cross-channel delle correnti ritardate. Non è una temporal self-attention. |
| `D`, dynamic TCAP | **Content-Gated Causal Multi-Lag Channel Mixer** | **gated CMLCM** | Variante del CMLCM; gating di ampiezza e allocazione sui ritardi dipendente dalle feature correnti. |
| `C0` modello completo | **compact spiking Transformer with pyramidal embedding, depthwise spatial mixing and causal multi-lag mixing** | nessuna sigla di modello finché non finale | Composizione di moduli: C0 non è sinonimo di TCAP. |
| `RI A`, intervalli senza relazione | **Causal interval-averaged memory** | — | Media della storia disponibile per bande di ritardo; gate di disponibilità, non necessariamente di contenuto. |
| `RI`, `RI-TCAP`, B−/B+ | **Interval-Relational Mixer** | **IRM** | Famiglia che usa memoria a intervalli e, nelle varianti bilineari, un termine relazionale sul presente. |
| `RI B−` | **IRM with low-rank antisymmetric temporal interaction** | **antisymmetric IRM** | Nome della variante; il termine bilineare è antisimmetrico, **non tutta la rete**. |
| `RI B+` | **IRM with low-rank symmetric temporal interaction** | **symmetric IRM** | Controllo a struttura/parametri comparabili. |
| `RI C` routed | **Content-gated interval-relational mixing** | **gated IRM** | Distinto dal routing D: differiscono memoria e interazione. |
| `RI points` | **Point-lag relational mixing** | — | Controllo che campiona posizioni ritardate, non medie di bande. |
| `RT`, memoria a tracce | **Relational Trace Mixer** | **RTM** | Famiglia: tracce di feature latenti con memoria normalizzata; non equivale a spike-trace STDP. |
| `RT static` | **Static-decay Relational Trace Mixer** | **static RTM** | Costanti di ritenzione addestrabili ma non condizionate dal dato al passo corrente. |
| `RT sensor` | **Sensor-activity-modulated Relational Trace Mixer** | **sensor-modulated RTM** | Hazard dipendente dai count locali degli eventi del sensore. |
| `RT latent` | **Latent-activity-modulated Relational Trace Mixer** | **latent-modulated RTM** | Hazard dipendente dall'attività spiking presinaptica interna, detached. |
| `RT content` | **Content-modulated Relational Trace Mixer** | **content-modulated RTM** | Hazard dipendente dalle feature correnti tramite router. |
| `ssa gain` | **Per-head SSA current scaling** | — | Controllo di riscalamento, non nuova famiglia di attenzione. |
| `ssa density_residual` | **Marginal-preserving decomposition of SSA current** | — | Scomposizione in termine atteso dai margini e residuo; non interpretarli come causa/rumore. |
| `M`, `R1` | **Temporal mean readout** / **spatiotemporal content-weighted readout** | — | Nomi funzionali, readout distinto dalla memoria. |
| `E0` | **Fixed-window, fixed-duration event-count frames** | E0 solo nelle tabelle di audit | 40 bin su finestra predefinita nelle coorti storiche; possibili code senza eventi. |
| `FEC` | **Fixed-event-count temporal binning** | FEC | Gruppi con quantità di eventi comparabile e durata fisica variabile. |
| `VC` | **Fixed-duration binning with variable sequence length** | VC solo se utile nelle tabelle | Δ unico calibrato sul training per dataset, sequenza dal primo all'ultimo evento, padding tecnico mascherato. |
| `M3` | **Intra-bin temporal moment features** | — | Non presentare come nuovo componente canonico della rete. |

**Nota sui nomi IRM e RTM:** si tratta di *etichette editoriali proposte* per i nostri moduli compositi. Le parole "interval memory", "trace", "exponential decay" e "antisymmetric bilinear interaction" descrivono nozioni preesistenti. Non rivendicarne la scoperta. Anche l'eventuale unicità delle sigle non è stata dimostrata.

## 3. Meccanismi che i nomi devono rappresentare davvero

### 3.1 CMLCM — ex TCAP statico

Sia \(z_t\) la **corrente convoluzionale pre-LIF**, non lo spike binario in uscita dal neurone:

\[
u_t=z_t+\sum_{k=1}^{K}W_k z_{t-d_k}.
\]

- Il contributo del presente ha percorso identità.
- Ogni \(W_k\) effettua mixing **cross-channel** al proprio ritardo \(d_k\), per ogni posizione spaziale.
- Le matrici sono inizializzate a zero nella versione storica.
- Le correnti e le trasformazioni non sono tutte spike-only; non dichiarare "only AC".
- Nei run C0 consolidati il CMLCM opera in due punti dei patch embedding, con tap finali \([1,2,4,8]\) nella coorte E0 di riferimento.
- "Causal" descrive **il filtro locale**, non garantisce causalità end-to-end nel training quando BatchNorm aggrega su più timestep.

La variante **gated CMLCM** ha \(u_t=z_t+\sum_k g_{k,t}W_k z_{t-d_k}\), con gate derivati dal contenuto presente; la parametrizzazione ampiezza/allocazione deve essere descritta quando rilevante.

### 3.2 IRM — ex RI-TCAP

La memoria causale considera bande \(I_k\) di ritardi e media soltanto le osservazioni disponibili:

\[
m_{k,t}=\frac{1}{n_{k,t}}\sum_{j\in I_k,\;t-j\;\mathrm{disponibile}}z_{t-j},
\]

con contributo nullo quando \(n_{k,t}=0\). Le frazioni di storia disponibile possono modulare un gate; non confonderlo con il router di contenuto.

Per la variante antisimmetrica a basso rango:

\[
r_{k,t}=(A m_{k,t})\odot(B z_t)-(B m_{k,t})\odot(A z_t),
\]
\[
u_t=z_t+\sum_k b_{k,t}\left(W_k m_{k,t}+U_k r_{k,t}\right).
\]

- La relazione cambia segno scambiando **i due argomenti del termine bilineare**; il classificatore nel suo insieme non diventa antisimmetrico.
- Le proiezioni \(A,B\) sono a rango ridotto; i percorsi lineari \(W_km_{k,t}\) restano.
- Nella campagna RI verificata il modulo relazionale sostituisce il mixer temporale **nel secondo embedding**; il primo mantiene il CMLCM.
- Non chiamare il modulo **attention** a meno di una derivazione esplicita che giustifichi tale interpretazione.

### 3.3 RTM — ex RT con tracce normalizzate

Il passato viene letto prima di inserire il presente. A ogni traccia corrispondono stato pesato \(M\) e massa \(w\):

\[
\widehat M_{k,t}=M_{k,t-1}/w_{k,t-1}\quad (w_{k,t-1}>0),
\]
\[
a_{k,t}=\exp(-h_{k,t}),\qquad
M_{k,t}=a_{k,t}M_{k,t-1}+(1-a_{k,t})X_t,
\]
\[
w_{k,t}=a_{k,t}w_{k,t-1}+(1-a_{k,t}).
\]

Se la massa è nulla, la lettura è zero. Nell'implementazione la stessa dinamica di aggiornamento è applicata anche alle proiezioni richieste dal termine relazionale; \(X_t\) indica quindi in maniera compatta le feature memorizzate e **non coincide necessariamente con il solo \(z_t\)**.

Nelle varianti sensor/latent, l'hazard segue il modello:

\[
h_{k,t}=h_{0,k}\left(e^{\theta_k}+\rho_k\,e_t/s\right),\quad \rho_k\ge0.
\]

- In **static RTM** il decadimento è appreso ma non dipende dal campione/istante.
- In **sensor-/latent-modulated RTM**, l'attività \(e_t\) aumenta l'hazard se \(\rho_k>0\); per costruzione questo **accorcia** la persistenza, non la estende. Eventuali effetti benefici sono da dimostrare.
- In **content-modulated RTM** un router sulle feature modifica il termine fisico. Non ricondurre erroneamente questa variante alla stessa formula additiva dell'attività.
- Le costanti di normalizzazione dell'attività sono calibrate sul training e conservate nel checkpoint; non si tarano sulla validation.
- Tracce di feature latenti **non sono** eligibility traces per apprendimento STDP; non rivendicare apprendimento locale né vera inferenza asincrona implementata.
- L'equivalenza sequence/chunk/step del *modulo* non basta a dimostrare streaming end-to-end del backbone.

### 3.4 Distinzione rispetto alla SSA

La self-attention implementata conserva il proprio significato di **spiking self-attention** (SSA). Un mixer temporale pre-LIF, un'interazione bilineare tra presente e passato, un readout con gate e un ridimensionamento della corrente SSA **non sono automaticamente varianti di self-attention**. Evitare di chiamare "attention" ogni gating o selezione temporale.

## 4. Come nominare la rete completa, quando i dati saranno conclusivi

Non assegnare oggi un brand o una sigla definitiva alla rete finale. Usare una descrizione compositiva che segua il checkpoint realmente selezionato:

- Se il riferimento finale resta **C0**: *compact spiking Transformer with causal multi-lag mixing*.
- Se viene promosso **IRM**: *compact spiking Transformer with interval-relational temporal mixing*.
- Se viene promosso **RTM**: *compact spiking Transformer with relational trace memory* (specificando se statica, sensor-, latent- o content-modulated).
- Se ci sono più contributi validati: descriverli separatamente, senza attribuire al nome un vantaggio aggregato mai testato.

Il nome distintivo del modello completo è **da decidere dopo** multi-seed, confronti controllati e verifica delle possibili collisioni semantiche e bibliografiche.

## 5. Terminologia sperimentale e figure

- Una tabella che lega un nome scientifico a un run deve riportare anche **ID originale, dataset, encoding, seed e riferimento alla config risolta**. Esempio: "antisymmetric IRM (internal: `VC_RI_M--interval-antisymmetric-r16`, seed 42)"; non rinominare il run.
- Scrivere *development-validation Macro-F1* o *official-test Macro-F1* esplicitamente; non chiamare "test accuracy" un risultato di validation.
- Evitare "adaptive" per il solo RTM statico: la costante è apprendibile fra training, ma l'hazard non è adattato all'input a ogni passo.
- Evitare "physical-time invariant": usare *fixed-duration input clock* e distinguere i parametri per passo LIF dalla scala fisica dei bin.
- Distinguere **buffer persistente di inferenza**, **attivazioni del backward**, **parametri**, **MAC**, **AC/SOP**, **stima di energia** e **energia misurata**.
- Per le tabelle d'ablazione usare etichette esplicative (*no interval-relational module*, *parameter-matched frame-local MLP*, *symmetric interaction*) più gli identificatori storici in nota.

## 6. Stato della verifica del prior art e metodo per finalizzarlo

Questa nomenclatura è stata formulata da codice e documentazione del progetto e da una **ricerca bibliografica preliminare**, non da una systematic review esaustiva. Sono già presenti in letteratura: self-attention spiking, meccanismi di delay/temporal interaction, filtri ricorrenti a decadimento esponenziale, trace-based learning e forme bilineari antisimmetriche. A titolo di confronto di funzione: QKFormer (NeurIPS 2024), *An Efficient Temporal Interaction Module for Spiking Transformer* (IJCAI 2024), *Temporal Interaction in Spiking Transformers with Multi-Delay Mixer* (CVPR 2026), STAtten (CVPR 2025); per trace-based plasticity si consultino lavori precedenti sulla STDP.

Prima di cristallizzare i nomi: verificare arXiv, IEEE Xplore, CVF, OpenReview, Scholar e GitHub con ricerca **semantica** su operatori e matematiche; registrare i lavori più vicini e indicare quali parti sono note, quali soltanto composte e quali effettivamente dimostrate. La mancata reperibilità di una stringa identica non costituisce prova di originalità.

## 7. Convenzione per modifiche future

Ogni nuovo modulo implementato deve avere una riga con: **ID tecnico → comportamento verificato nel codice → denominazione descrittiva → stato sperimentale → riferimenti**. Se una variante non entra nel modello finale, la sua denominazione può restare nel registro tecnico senza essere introdotta nella tesi principale. Le modifiche dei nomi scientifici devono essere applicate coerentemente a testo, equazioni, figure e tabelle, **mai retroattivamente agli artifact**.

**Fonti interne di riferimento:** `EVOLUZIONE_DELLA_SPERIMENTAZIONE_DETTAGLIATA.md` (stato 1/11/2026), `EXPERIMENT_LEDGER.md`, configurazioni risolte dei run; nel repository del codice: `src/etsr/models/temporal.py`, `interval.py`, `adaptive_memory.py`, `attention_current.py`, `mini_qkformer.py`. Verificare il branch e lo SHA attuali prima di finalizzare una descrizione.
