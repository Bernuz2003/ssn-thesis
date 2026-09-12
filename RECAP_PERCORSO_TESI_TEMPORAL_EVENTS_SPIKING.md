# Recap completo del percorso di tesi — Temporal-Events-Spiking

> **Documento di lavoro — stato del progetto al branch `developer`**
>
> Repository analizzato: `Bernuz2003/Temporal-Events-Spiking`  
> Branch principali considerati:
>
> - `refactor/mechanistic-temporal-audit`: fase di ispezione meccanicistica della rappresentazione temporale della baseline;
> - `developer`: fase successiva di innovazione architetturale guidata principalmente da DVS-Lip.
>
> Questo documento ricostruisce il **percorso scientifico**, non una cronologia di commit. L'obiettivo è rendere chiara la logica che collega ipotesi, esperimenti, risultati, decisioni progettuali e stato attuale del lavoro.

---

## 1. Idea generale del lavoro di tesi

Il filo conduttore della tesi è la progettazione di uno **Spiking Transformer compatto, realmente adatto a dati neuromorfici temporali e compatibile con un futuro deployment hardware**, con particolare attenzione a FPGA e a scenari edge.

Il punto di partenza non è stato “aggiungere componenti” a un Transformer in modo incrementale, ma affrontare una domanda più fondamentale:

> **Una Mini-QKFormer compatta, pur essendo una SNN e pur mantenendo stato di membrana nel tempo, sta davvero utilizzando in modo efficace la struttura temporale fine degli stream DVS?**

Questa domanda è particolarmente importante perché nei dati da Dynamic Vision Sensor la dimensione temporale non è un dettaglio accessorio. Gli eventi sono prodotti in risposta a variazioni locali di luminanza e arrivano con risoluzione temporale molto più fine rispetto ai frame convenzionali. Il contenuto discriminativo può quindi risiedere non soltanto in **quali pattern spaziali** compaiono, ma anche in:

- ordine temporale delle primitive;
- velocità di evoluzione;
- dipendenze tra eventi distanti;
- persistenza di informazione dopo la fine dell'attività sensoriale;
- burstiness e densità locale degli eventi;
- variazioni rapide che rischiano di essere perse dal binning;
- correlazioni tra canali e tra scale temporali.

Da qui nasce una strategia a due grandi fasi già percorse:

1. **prima capire il comportamento temporale della baseline**, attraverso un audit meccanicistico controllato;
2. **poi modificare l'architettura**, ma soltanto con cambiamenti motivati da evidenza empirica e con un protocollo che separi il guadagno architetturale dal semplice aumento di capacità o dal tuning.

La seconda fase ha progressivamente identificato due modifiche particolarmente solide:

- **F**, cioè il frontend piramidale più efficiente;
- **TCAP**, implementato nel codice come *causal temporal channel mixer*, cioè un meccanismo esplicito di mixing temporale causale tra canali.

La loro combinazione, **F+TCAP**, è al momento il riferimento prestazionale principale del lavoro, con un miglioramento netto rispetto alla baseline pur restando nel budget compatto.

---

# PARTE I — FASE DI ISPEZIONE MECCANISTICA DELLA TEMPORALITÀ

## 2. Perché partire da un mechanistic temporal audit

La baseline della tesi deriva concettualmente da QKFormer, ma viene fortemente ridimensionata per ottenere un modello nell'ordine di circa mezzo milione di parametri.

Una SNN di questo tipo possiede già dinamiche temporali implicite attraverso i neuroni LIF: il potenziale di membrana evolve da un timestep al successivo e costituisce una forma di memoria causale. Tuttavia, questo **non implica automaticamente** che il modello apprenda rappresentazioni temporali semanticamente robuste.

Esistono almeno tre possibilità molto diverse:

1. il modello può utilizzare davvero l'ordine temporale;
2. il modello può riconoscere principalmente il contenuto spaziale e usare il tempo soltanto come debole segnale ausiliario;
3. il modello può essere sensibile al tempo, ma in modo fragile, dipendente da shortcut statistiche o incapace di conservare correttamente l'informazione nei layer profondi.

Passare direttamente all'innovazione architetturale avrebbe reso difficile interpretare gli esperimenti successivi. Se una modifica avesse migliorato l'accuracy, non sarebbe stato chiaro se stesse correggendo un deficit temporale, aumentando semplicemente la capacità o introducendo una scorciatoia statistica.

Per questo il branch `refactor/mechanistic-temporal-audit` è stato concepito come una fase diagnostica indipendente.

---

## 3. Dataset diagnostico: DVS-Gesture-Chain order-2

Per studiare il concetto di **ordine**, è stato utilizzato un dataset diagnostico basato su concatenazioni controllate di primitive DVS, indicato nella documentazione come **DVS-Gesture-Chain order-2 / DVSGC**.

L'idea è semplice ma potente: se due sequenze contengono le stesse primitive `A` e `B`, ma in ordine opposto,

\[
A \rightarrow B
\qquad \text{e} \qquad
B \rightarrow A,
\]

allora il contenuto globale può essere quasi identico, mentre cambia il significato temporale.

Questo consente di fattorizzare due capacità che in un normale benchmark tendono a essere mischiate:

- **Content Recognition**: il modello ha riconosciuto quali primitive sono presenti?
- **Order Recognition**: dato che il contenuto è riconosciuto, il modello ha riconosciuto l'ordine corretto?

Questa separazione è cruciale: un'accuracy di classificazione apparentemente elevata potrebbe nascondere una rappresentazione temporale debole.

---

## 4. Prima esplorazione comportamentale

La prima fase dell'audit ha utilizzato perturbazioni temporali semplici ma informative. Il notebook archiviato in:

```text
notebooks/archive/dvsgc/mechanistic_audit_analysis.ipynb
```

riporta, tra gli altri, questi risultati esplorativi:

| Condizione | Content Accuracy | Conditional Order Accuracy |
|---|---:|---:|
| Originale | **89.58%** | **88.37%** |
| Temporal shuffle | **90.97%** | **41.98%** |
| Reverse time | **88.89%** | **10.16%** |
| Reverse actions mantenendo il target originale | **89.58%** | **11.63%** |

Il pattern è estremamente informativo.

### 4.1 Il contenuto sopravvive alle trasformazioni temporali

La Content Accuracy resta circa nell'intervallo 89–91%. Questo significa che il modello continua a riconoscere *quali* primitive sono presenti anche quando l'ordine viene fortemente alterato.

La rappresentazione di contenuto è quindi relativamente robusta.

### 4.2 L'ordine crolla quando viene distrutto o invertito

La Conditional Order Accuracy passa da circa 88% a:

- circa 42% con shuffle temporale;
- circa 10% con time reversal;
- circa 12% invertendo semanticamente l'ordine ma mantenendo il target originale.

Questo dimostra che la rete **non è temporalmente cieca**.

Il modello usa informazione temporale reale. Se fosse soltanto un riconoscitore quasi statico del contenuto, il cambio di ordine non avrebbe prodotto un collasso così marcato.

### 4.3 Prima conclusione, volutamente limitata

L'evidenza iniziale giustificava una conclusione moderata:

> Mini-QKFormer usa l'ordine temporale, ma la rappresentazione dell'ordine sembra meno robusta della rappresentazione del contenuto.

Non era ancora possibile dire:

- dove l'ordine venisse codificato;
- se venisse perso lungo la gerarchia;
- se il problema fosse nel frontend, nell'attention, nei LIF o nel readout;
- se le perturbazioni stessero introducendo shortcut o distribuzioni anomale.

Per questo l'audit è stato successivamente rafforzato.

---

## 5. Perché la prima analisi non era sufficiente

Un risultato importante del percorso metodologico è stato riconoscere che una singola perturbazione su un singolo checkpoint non è una base sufficiente per una conclusione meccanicistica forte.

Sono emersi diversi rischi:

- dipendenza da un singolo seed;
- confusione tra effetto di contenuto ed effetto di ordine;
- possibilità che una trasformazione alteri statistiche di basso livello oltre alla semantica temporale;
- selezione implicita del checkpoint;
- rischio di leggere troppo in una differenza tra due prefissi temporali;
- uso iniziale del test ufficiale, che rendeva inopportuno continuare a usarlo per guidare la ricerca.

La risposta non è stata “fare più perturbazioni”, ma progettare un protocollo più rigoroso.

---

# 6. Temporal Utilization Diagnostic Profile

L'audit viene quindi riformulato come un **Temporal Diagnostic / Utilization Profile**, con l'obiettivo di separare più domande meccanicistiche.

Il protocollo definitivo mira a caratterizzare:

1. **fattorizzazione contenuto/ordine**;
2. **coerenza sotto trasformazioni temporali semantiche**;
3. **utilità dell'evidenza aggiunta nel tempo**;
4. **disponibilità layer-wise dell'informazione temporale**;
5. **uso causale di tale informazione**;
6. **possibili shortcut statistiche di input**.

Questa è una scelta metodologica importante per la tesi: non viene cercato un singolo “temporal score”, perché un solo numero mescolerebbe fenomeni differenti.

---

## 7. Split rigoroso e embargo del test

Dopo la fase esplorativa, il protocollo meccanicistico utilizza esclusivamente il training ufficiale, suddiviso in:

- **70% `train_core`**;
- **15% `checkpoint_validation`**;
- **15% `development_audit`**.

Lo split è:

- deterministico;
- raggruppato per `source_filename`;
- progettato per mantenere le coppie inverse nello stesso split.

Il test ufficiale non viene più utilizzato per guidare le decisioni.

Questa decisione anticipa una filosofia che verrà poi applicata in modo ancora più netto su DVS-Lip: **il test ufficiale è una risorsa finale, non uno strumento di ricerca iterativa**.

---

## 8. Dataset matched e coppie inverse esatte

Per ridurre i confondenti viene costruito un dataset matched in:

```text
data/matched_dvsgc_order2_v1/
```

con:

- 16 frame temporali;
- 13 frame per le primitive di base;
- primitive `1`, `3`, `8`;
- pairing esatto AB/BA;
- stessi chunk primitivi;
- stesse durate;
- boundary temporali noti;
- identificativo di coppia;
- manifest e hash per la riproducibilità.

Questa costruzione consente di chiedere una cosa molto più precisa:

> Se al modello presento esattamente gli stessi “ingredienti”, cambiando soltanto l'ordine semantico, la predizione si trasforma in modo coerente?

---

## 9. Inverse Temporal Consistency

Viene introdotta la **Inverse Temporal Consistency (ITC)**.

Per una coppia matched `AB` e `BA`, si verifica se la predizione della seconda sequenza corrisponde esattamente all'inversa semantica della predizione sulla prima.

In forma concettuale:

\[
\hat y(BA) = \rho(\hat y(AB)),
\]

dove \(\rho\) è l'operatore che mappa una classe ordinata nella sua classe inversa.

L'ITC è diversa dalla semplice accuracy:

- non chiede necessariamente che `AB` sia classificato correttamente;
- misura se il comportamento del modello **trasforma coerentemente** al trasformarsi dell'ordine.

Nella prima analisi paired l'ITC aggregata era nell'ordine del **63.89%**, evidenza che la rete possiede una sensibilità trasformazionale all'ordine, ma non una perfetta equivarianza temporale.

---

## 10. Robustezza multi-seed

Il protocollo definitivo utilizza tre seed:

```text
42, 123, 2026
```

e richiede, per considerare un effetto strutturale:

- stesso segno dell'effetto tra seed;
- assenza di dipendenza dominante da un singolo checkpoint;
- coerenza con l'analisi content-pair;
- sopravvivenza ai controlli.

Questo è un passaggio concettualmente importante: l'obiettivo dell'audit non è produrre la massima accuracy, ma distinguere **fenomeni reali della rappresentazione** da accidenti di training.

---

## 11. Differenze tra coppie di primitive

Nel notebook multi-seed, la difficoltà di riconoscimento del contenuto varia in modo non trascurabile tra coppie:

| Coppia | Content Accuracy media | Conditional Order Accuracy media |
|---|---:|---:|
| `1|3` | **66.67% ± 10.05** | **93.82% ± 5.50** |
| `1|8` | **89.58% ± 3.61** | **88.33% ± 5.55** |
| `3|8` | **66.67% ± 4.77** | **92.18% ± 2.73** |

Questa osservazione modifica l'interpretazione ingenua della baseline.

Non emerge uno scenario del tipo “la rete non sa l'ordine”. Al contrario:

> **quando il contenuto è riconosciuto, l'ordine condizionato è generalmente molto forte; parte della difficoltà nasce dalla discriminabilità delle primitive stesse.**

Questo è un esempio concreto del valore della fattorizzazione contenuto/ordine: un'unica accuracy avrebbe mescolato i due fenomeni.

---

# 12. Prefix analysis: quando emerge l'informazione utile?

Il modello viene valutato su prefissi progressivi della sequenza.

L'obiettivo non è soltanto misurare l'accuracy a tempo finale, ma capire **come si costruisce la decisione**.

All'inizio della sequenza le metriche sono naturalmente basse. Per esempio, nel notebook multi-seed:

- timestep 1: accuracy ≈ **29.86% ± 2.62**;
- timestep 4: accuracy ≈ **41.32% ± 4.21**.

L'informazione discriminativa si accumula nel tempo, come atteso per una rete che deve distinguere sequenze ordinate.

Il protocollo salva:

- logit per timestep;
- logit cumulativi di prefisso;
- accuracy di prefisso;
- metriche contenuto/ordine;
- AUC dei prefissi.

Questo modo di analizzare il modello verrà successivamente ripreso, con finalità diverse, su DVS-Lip per distinguere modelli che hanno la stessa prestazione finale ma dinamiche temporali molto diverse.

---

# 13. Late Harm, Late Rescue e Net Late Utility

Una delle prime osservazioni esplorative sembrava indicare che l'ultima parte della sequenza potesse essere **distruttiva**: in un checkpoint, le predizioni corrette diminuivano tra il prefisso `12/16` e `16/16`.

Il valore esplorativo corrispondeva a una Net Late Utility negativa di circa:

\[
-6.94 \text{ punti percentuali}.
\]

Questo era un segnale interessante, ma non sufficientemente robusto.

Il protocollo definitivo introduce tre quantità:

- **Late Harm**: campioni corretti prima dell'ultima evidenza e sbagliati dopo;
- **Late Rescue**: campioni sbagliati prima e corretti dopo;
- **Net Late Utility**:

\[
\mathrm{NLU} = \mathrm{LateRescue} - \mathrm{LateHarm}.
\]

La verifica multi-seed del tratto `12/16 → 16/16` mostra invece una **Net Late Utility positiva, circa +7.64 pp**, con intervallo bootstrap aggregato positivo.

Questa è una delle conclusioni metodologicamente più importanti dell'intera fase:

> **la prima apparente evidenza di “late evidence distruttiva” non era un fenomeno robusto della baseline.**

Il risultato corretto non è stato forzato per confermare l'ipotesi iniziale: l'audit più rigoroso ha **confutato** quella lettura.

Per la tesi questo passaggio è prezioso perché mostra un workflow di ricerca corretto: un'anomalia iniziale genera un'ipotesi, ma l'ipotesi viene sottoposta a controlli più forti e, se non regge, viene abbandonata.

---

## 14. Probe lineari layer-wise

Per capire **dove** sia disponibile l'informazione, la baseline viene strumentata con:

```python
forward_with_trace(...)
```

senza alterare il normale forward di training.

Sono estratte feature da:

```text
patch_embed1
stage1
patch_embed2
stage2
```

e vengono addestrati probe lineari frozen-backbone per predire:

- classe completa;
- primitiva corrente;
- primitiva precedente;
- ordine.

I probe vengono valutati a timestep selezionati, tra cui:

```text
4, 8, 12, 16
```

con:

- fitting su `train_core`;
- regolarizzazione scelta su `checkpoint_validation`;
- misura finale su `development_audit`;
- standardizzazione appresa esclusivamente dal train;
- controllo con label permutate.

Lo scopo non è migliorare il modello, ma separare due concetti:

- **informazione disponibile** nella rappresentazione;
- **informazione realmente utilizzata** dalla rete per la predizione.

---

## 15. Activation patching causale

Per superare il limite dei probe puramente decodificativi viene introdotto il **causal activation patching**.

Per una sequenza `AB`, si prende come donor la sequenza matched `BA` e si sostituisce una traiettoria interna in una regione semanticamente allineata.

Vengono patchate regioni corrispondenti a:

- prima azione;
- seconda azione.

La sostituzione è segment-aligned e non attraversa i boundary.

La quantità primaria è la variazione del margine tra classe corretta e classe inversa:

\[
\Delta m =
[z_y - z_{\rho(y)}]_{\text{patched}}
-
[z_y - z_{\rho(y)}]_{\text{baseline}}.
\]

Sono inoltre misurati:

- prediction-change rate;
- inverse-class prediction rate;
- cambiamenti specificamente verso la classe inversa;
- effetto per layer e per regione.

L'interpretazione prevista dal protocollo è:

> un'informazione è candidata a essere **causalmente usata** se è decodificabile da un probe e un patch semanticamente mirato produce un effetto coerente sul margine.

Questa distinzione è fondamentale: un'informazione può essere presente in un hidden state ma completamente ignorata dal readout downstream.

---

## 16. Controlli sulle shortcut di input

Il protocollo salva per ogni timestep:

- conteggio eventi totale;
- conteggio per polarità;
- rapporto di polarità;
- variazione L1 dal frame precedente;
- centroide spaziale;
- segmento semantico.

Vengono confrontate due baseline:

1. statistiche temporali che mantengono l'ordine;
2. statistiche invarianti alla permutazione temporale.

Lo scopo è verificare se l'ordine possa essere predetto da banali differenze di:

- event count;
- polarità;
- movimento del centroide;
- durata.

Un classificatore semplice capace di risolvere il task tramite queste statistiche avrebbe ridotto la forza dell'interpretazione “semantica” dei risultati della rete.

---

## 17. Risultato concettuale della fase meccanicistica

La fase non porta alla conclusione semplicistica “Mini-QKFormer non modella il tempo”.

Al contrario, il quadro più corretto è:

- la rete **usa realmente l'ordine temporale**;
- il contenuto è in parte più robusto e più facilmente riconoscibile;
- la difficoltà varia tra primitive;
- la sensitivity all'inversione temporale è forte;
- l'evidenza finale non è globalmente distruttiva quando il fenomeno viene misurato in modo robusto multi-seed;
- il tempo è già presente nella computazione, ma ciò non implica che il meccanismo temporale sia ottimale per benchmark più complessi.

L'audit quindi non “dimostra che serve un nuovo modulo”, ma giustifica una domanda più matura:

> **quanto può migliorare una Mini-QKFormer compatta se, oltre alla memoria implicita LIF, introduciamo meccanismi architetturali espliciti e controllati per rappresentare dipendenze spazio-temporali?**

Questa domanda porta direttamente alla fase `developer`.

---

# PARTE II — PASSAGGIO A DVS-LIP E RICERCA ARCHITETTURALE CONTROLLATA

## 18. Perché DVS-Lip diventa il benchmark di sviluppo principale

DVS-Lip è un benchmark particolarmente adatto alla seconda fase perché il riconoscimento labiale event-based richiede:

- micro-movimenti;
- coordinazione temporale fine;
- pattern che evolvono per centinaia di millisecondi;
- integrazione su scala lunga;
- capacità di distinguere configurazioni spaziali simili con evoluzione differente.

Inoltre, il modello deve restare estremamente compatto.

Quindi DVS-Lip offre un contesto più difficile e più vicino all'obiettivo della tesi rispetto al dataset diagnostico DVSGC.

---

# 19. Protocollo sperimentale DVS-Lip

La validazione di sviluppo contiene:

> **2,995 campioni**

estratti esclusivamente dal training ufficiale.

Il **test ufficiale resta embargato**.

Salvo diversa indicazione, il protocollo usa:

- seed `42`;
- finestra fisica di **2 s**;
- **40 bin × 50 ms**;
- readout mean su finestra fissa;
- ricetta comune derivata da `dvslip_e0_128`;
- selezione checkpoint sulla Macro-F1 della stessa dev validation.

Questa ultima scelta va interpretata correttamente: i numeri attuali sono **evidenza di sviluppo**, non ancora una stima finale indipendente della generalizzazione.

---

# 20. Baseline B: Mini-QKFormer

La baseline compatta viene indicata come **B**.

Configurazione concettuale:

```text
input DVS binned
→ patch embedding SPEDS-like
→ stage1 con Q-K token attention
→ patch embedding stage2
→ stage2 con spiking self-attention
→ spatial mean
→ temporal mean
→ classifier
```

Dimensione:

> **500,708 parametri**

Prestazione di riferimento:

- Accuracy: **44.81%**
- Macro-F1: **44.15%**
- Acc@1s: **37.07%**
- Acc@2s: **52.54%**
- Prefix-AUC: **0.2453**

Questa baseline diventa il controllo contro cui ogni modifica strutturale viene confrontata.

---

# 21. Architettura interna della baseline

Nel file:

```text
src/etsr/models/mini_qkformer.py
```

il modello è descritto come:

> “Compact QKFormer-like DVS-Lip baseline”

e la documentazione specifica correttamente che non viene rivendicata una riproduzione bit-exact di QKFormer.

I blocchi principali sono implementati in:

```text
src/etsr/models/layers.py
```

### 21.1 Stage 1: Q-K Token Attention

La token attention utilizza Q e K spike-form e produce un vettore binario di importanza dei token.

Concettualmente:

\[
q, k \in \{0,1\}^{T \times N \times D}
\]

e viene ricavato uno score di token tramite riduzione sui canali/head, quindi una LIF produce il mask binario.

Non viene costruita una classica matrice d'attenzione dense \(N \times N\).

Questo è coerente con il vincolo hardware della tesi: mantenere operazioni spike-driven e complessità contenuta.

### 21.2 Stage 2: Spiking Self-Attention

Lo stage profondo utilizza Q, K, V spike-form e una forma efficiente del prodotto associativo:

\[
Q(K^\top V),
\]

seguita da dinamica spiking e proiezione.

### 21.3 Memoria LIF

Il neurone di base implementa soft reset:

\[
u_t
=
u_{t-1}
+
\frac{x_t-u_{t-1}}{\tau},
\]

\[
s_t = H(u_t-\theta),
\]

\[
u_t \leftarrow u_t - s_t\theta.
\]

Durante il training, \(H\) usa forward hard e surrogate gradient nel backward.

Questa dinamica costituisce la memoria temporale implicita già presente nella baseline.

---

# 22. Il primo controllo indispensabile: la capacità

Prima di attribuire i limiti della baseline a un difetto meccanicistico, vengono addestrate versioni più grandi.

| Modello | Parametri | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| B | 500,708 | 44.81% | 44.15% |
| Capacity ~1M | 1,113,508 | 49.58% | 49.38% |
| Capacity ~2M | 1,967,972 | 51.79% | 51.81% |

Il risultato dice chiaramente che:

> **il task non è saturato dalla baseline.**

Aumentare capacità migliora molto.

Ma queste versioni non sono candidati finali: servono come **upper control**.

Diventa quindi possibile giudicare una nuova architettura non soltanto contro B, ma anche rispetto alla domanda:

> può ottenere un guadagno simile a un modello 2–4 volte più grande restando circa a 500K?

---

# 23. NoCrossTime: quanto conta davvero la ricorrenza temporale?

Viene introdotta una variante in cui la propagazione della membrana tra bin è disabilitata.

Risultato:

- Accuracy: **15.13%**
- Macro-F1: **13.01%**

cioè circa:

> **−31.14 pp Macro-F1 rispetto a B.**

Questo è un controllo estremamente importante.

Dimostra che, su DVS-Lip:

> **la dipendenza cross-time è indispensabile.**

Quindi la rete non può essere reinterpretata come un modello quasi-frame-based che beneficia soltanto di aggregazione finale.

Allo stesso tempo, il risultato non dice che il LIF standard sia sufficiente: dice soltanto che rimuovere la memoria distrugge il task.

---

# 24. Readout temporale: il problema non è semplicemente “mediare troppo a lungo”

Sono testate varianti che leggono la sequenza in corrispondenza dell'ultimo evento:

| Readout | Accuracy | Macro-F1 |
|---|---:|---:|
| B fixed-window mean | 44.81% | 44.15% |
| mean @ last event | 42.50% | 42.02% |
| last @ last event | 32.32% | 31.83% |

Nessuna migliora la baseline.

Questa evidenza chiude una spiegazione troppo semplice:

> non basta tagliare il silenzio post-evento o cambiare il readout per risolvere il limite.

---

# PARTE III — F: FRONTEND PIRAMIDALE

## 25. Motivazione

Il frontend baseline mantiene una geometria relativamente costosa in termini di:

- stati LIF;
- attivazioni intermedie;
- letture/scritture di memoria;
- synaptic operations.

Su hardware neuromorfico/FPGA il numero di parametri è solo una delle componenti del costo. Il footprint degli **stati di membrana** e il traffico necessario per aggiornarli possono essere altrettanto importanti.

L'ipotesi di F è quindi:

> **comprimere più aggressivamente la rappresentazione spaziale nei layer iniziali, senza eliminare la struttura gerarchica e senza modificare il resto del protocollo, può migliorare contemporaneamente capacità utile, generalizzazione e costo hardware.**

---

# 26. Implementazione di F

In `layers.py`, `PyramidalPatchEmbedding` sostituisce il frontend iniziale.

La progressione dei canali è:

\[
D/16
\rightarrow
D/8
\rightarrow
D/4
\rightarrow
D/2.
\]

Le riduzioni avvengono progressivamente attraverso blocchi:

```text
Conv → BN → MaxPool → LIF
```

con shortcut spike-driven dalla seconda rappresentazione verso l'output tramite:

```text
1×1 convolution, stride 4
```

La geometria viene quindi ridotta **prima**, evitando di mantenere feature map grandi e stati LIF ampi fino a layer relativamente profondi.

Questo è concettualmente vicino a un principio piramidale/hierarchical representation: alta risoluzione nei primi passi, compressione progressiva, maggiore capacità semantica sui feature map più piccoli.

---

# 27. Risultato di F

F ottiene:

- **431,076 parametri**
- Accuracy: **46.88%**
- Macro-F1: **46.15%**

Guadagno su B:

- **+2.07 pp Accuracy**
- **+1.99 pp Macro-F1**

con:

> **−13.91% parametri**

Quindi F è uno dei primi risultati realmente Pareto-positive del progetto.

---

# 28. Profilo hardware di F

Il profiler mostra:

| Metrica | B | F | Delta |
|---|---:|---:|---:|
| Firing rate | 0.0585 | 0.1352 | aumenta |
| Stateful elements | 1,099,776 | 477,184 | **−56.61%** |
| State reads | 42.89M | 18.61M | forte riduzione |
| State writes | 45.09M | 19.56M | forte riduzione |
| Potential AC | 5,557.45M | 2,904.56M | **−47.64%** |
| Multi-value MAC | 1,530.93M | 1,247.82M | **−18.49%** |
| Horowitz proxy activity | 7,319.57 µJ | 6,191.70 µJ | **−15.41%** |
| Horowitz proxy dense | 12,058.41 µJ | 8,368.47 µJ | **−30.60%** |

Il punto più interessante è che la firing rate **aumenta**.

Quindi il vantaggio energetico non deriva da una sparsity maggiore.

Deriva da una struttura migliore:

> **meno siti neuronali da mantenere e meno operazioni potenziali per timestep.**

Questa è una conclusione molto importante per il futuro FPGA:

> una rete può essere più efficiente anche se spike-a di più, se la geometria su cui spike-a è sufficientemente più piccola.

---

# PARTE IV — TCAP: CAPACITÀ TEMPORALE ESPLICITA

## 29. Motivazione

La baseline ha già memoria attraverso i LIF, ma tale memoria è principalmente **per-neurone**.

Un LIF standard aggiorna il proprio stato nel tempo, ma non introduce esplicitamente una trasformazione tra canali provenienti da ritardi differenti.

La nuova ipotesi è:

> **DVS-Lip può richiedere non soltanto persistenza temporale, ma relazioni temporali cross-channel esplicite e multi-delay.**

Il codice implementa questa idea attraverso:

```text
CausalTemporalChannelMixer
```

che nella documentazione degli esperimenti viene indicato come **TCAP**.

---

# 30. Formulazione di TCAP

Per ritardi:

\[
\mathcal{D}=\{1,2,4\},
\]

il modulo calcola:

\[
y_t =
x_t +
\sum_{\delta \in \mathcal D}
W_\delta x_{t-\delta},
\]

con:

\[
W_\delta \in \mathbb{R}^{C \times C}.
\]

Le proprietà fondamentali sono:

- **causale**: nessuna informazione futura;
- **multi-delay**: più orizzonti temporali;
- **cross-channel**: i canali passati possono contribuire a canali differenti al tempo corrente;
- **identity-preserving all'inizializzazione**: i \(W_\delta\) sono inizializzati a zero.

L'ultimo punto è particolarmente importante per il training.

All'inizio:

\[
y_t = x_t,
\]

quindi il nuovo modello parte esattamente dal comportamento del backbone locale anziché da una trasformazione temporale casuale.

---

# 31. Dove viene applicato

TCAP viene inserito in punti selezionati dei patch embedding, non come attention temporale globale \(T \times T\).

Questo evita:

- softmax temporale;
- matrice quadratica;
- memoria di attenzione su tutti i timestep;
- dipendenza non causale.

Il costo cresce principalmente con:

\[
O(|\mathcal D| C^2 N),
\]

ma lo stato richiesto è limitato agli ultimi `max(delay)` timestep.

Per FPGA è quindi molto più realistico di una self-attention globale nel tempo.

---

# 32. B+TCAP

Applicato alla baseline B:

- parametri: **562,148**
- Accuracy: **48.38%**
- Macro-F1: **48.12%**
- Acc@1s: **40.95%**
- Acc@2s: **55.81%**
- Prefix-AUC: **0.2611**

Guadagno su B:

- **+3.57 pp Accuracy**
- **+3.97 pp Macro-F1**

Il risultato è importante perché TCAP migliora più di F quando testato isolatamente.

Dimostra che un temporal mixer esplicito aggiunge capacità utile che il solo LIF standard non possiede.

---

# 33. Costo hardware di B+TCAP

TCAP non è gratuito:

- firing rate: **0.0468**, quindi inferiore a B;
- activity AC: **198.14M**, inferiore a B;
- ma multi-value MAC: **1,782.59M**, cioè circa **+16.44%**;
- Horowitz activity proxy: **8,392.66 µJ**, cioè **+14.66%**.

Questo porta a una distinzione necessaria:

> una firing rate più bassa non implica automaticamente energia inferiore.

TCAP riduce spike activity ma introduce trasformazioni cross-channel multi-value.

Quindi il suo valore è principalmente **prestazionale/temporale**, e deve essere compensato da una geometria più efficiente.

Questa considerazione rende naturale la combinazione con F.

---

# PARTE V — F+TCAP: IL RISULTATO CENTRALE ATTUALE

## 34. Ipotesi di complementarità

F e TCAP agiscono su due colli di bottiglia diversi:

- **F** migliora la rappresentazione spaziale gerarchica e riduce il costo del frontend;
- **TCAP** aggiunge capacità temporale causale esplicita.

Se i due guadagni fossero dovuti allo stesso effetto, la combinazione potrebbe saturare o peggiorare.

Se invece fossero complementari, dovrebbe emergere un guadagno ulteriore.

---

# 35. Risultato

**F+TCAP** ottiene:

- parametri: **492,516**
- Accuracy: **52.79%**
- Macro-F1: **52.36%**
- Acc@1s: **44.42%**
- Acc@2s: **61.15%**
- Prefix-AUC: **0.2916**
- peak GPU: circa **3.50 GiB**
- tempo full run: circa **6.61 h**

Guadagno rispetto a B:

- **+7.98 pp Accuracy**
- **+8.21 pp Macro-F1**

con un numero di parametri:

> **inferiore alla baseline originale.**

Questo è al momento il risultato strutturale più forte dell'intero percorso.

---

# 36. Perché F+TCAP è scientificamente più interessante del capacity scaling

Il controllo ~2M raggiunge:

- Accuracy: **51.79%**
- Macro-F1: **51.81%**

F+TCAP raggiunge:

- Accuracy: **52.79%**
- Macro-F1: **52.36%**

Quindi, con circa:

\[
0.493\text{M parametri},
\]

supera un modello da quasi:

\[
1.97\text{M parametri}.
\]

Questo supporta la tesi secondo cui il problema non era soltanto “Mini-QKFormer è troppo piccola”.

Una parte consistente del gap viene recuperata attraverso una **migliore allocazione della capacità**:

- più gerarchia utile nel frontend;
- memoria temporale cross-channel esplicita.

---

# 37. Evidenza paired rispetto a B e F

Sui medesimi 2,995 campioni:

### B vs F

- B corretti: **1,342**
- F corretti: **1,404**
- F corregge **452** errori di B;
- F introduce **390** errori su campioni corretti da B.

### B vs TCAP

- TCAP corretti: **1,449**
- TCAP corregge **488** errori di B;
- introduce **381** nuovi errori.

### F vs TCAP

- TCAP corregge **445** errori di F;
- F corregge **400** errori di TCAP.

Questo indica che F e TCAP non stanno semplicemente migliorando gli stessi esempi.

È ulteriore evidenza della loro complementarità.

---

# 38. Significatività esplorativa

Sulla stessa validation:

- McNemar F vs B: \(p \approx 0.0355\)
- McNemar TCAP vs B: \(p \approx 0.000318\)
- McNemar PLIF vs B: \(p \approx 0.287\)
- McNemar TCAP vs F: \(p \approx 0.130\)

Bootstrap stratificato per:

\[
\Delta F1(\text{TCAP} - \text{F})
\]

fornisce:

- differenza ≈ **+1.97 pp**
- CI 95% ≈ **[0.02, 3.87]**

Questi test sono utili per prioritizzare le direzioni, ma non sono ancora inferenza confermativa perché:

- i checkpoint sono stati selezionati sulla stessa validation;
- la ricerca è guidata da un singolo seed;
- sono state provate più configurazioni.

La verifica finale richiederà nuovi seed comuni.

---

# PARTE VI — PLIF: UN ESPERIMENTO NEGATIVO MA MOLTO INFORMATIVO

## 39. Ipotesi PLIF

PLIF rende apprendibile la costante temporale del neurone.

Nel codice il parametro è rappresentato attraverso una parametrizzazione stabile dell'inverse tau; concettualmente:

\[
\tau_c = \frac{1}{\sigma(\alpha_c)},
\]

con un valore apprendibile per canale.

L'ipotesi è che canali differenti possano richiedere differenti horizon temporali.

---

# 40. Prestazione finale

B+PLIF:

- parametri: **502,676**
- Accuracy: **43.74%**
- Macro-F1: **43.68%**

quindi:

> **−0.47 pp Macro-F1 rispetto a B.**

Come candidato finale, PLIF non supera il gate.

Ma fermarsi alla metrica finale sarebbe un errore.

---

# 41. La dinamica temporale di PLIF è molto diversa

F1 ai prefissi:

| Tempo | B | F | TCAP | PLIF |
|---|---:|---:|---:|---:|
| 250 ms | 1.41 | 1.08 | 0.57 | 1.66 |
| 500 ms | 5.15 | 5.29 | 3.70 | 3.93 |
| 1.0 s | 20.94 | 16.85 | 22.62 | **27.60** |
| 1.5 s | 30.44 | 29.28 | 36.25 | **38.76** |
| 2.0 s | 44.15 | 46.15 | **48.12** | 43.68 |

PLIF domina nettamente tra 1 e 1.5 secondi, poi viene superato.

Prefix-F1 AUC migliora di circa:

> **+3.47 pp**

rispetto a B.

Questo mostra che PLIF non è semplicemente “un B peggiore”.

Ha una **diversa dinamica di memoria**.

---

# 42. Durata reale dei campioni DVS-Lip

Distribuzione approssimativa:

- media: **1.083 s**
- mediana: **1.079 s**
- P75: **1.184 s**
- P95: **1.348 s**
- P99: **1.457 s**
- entro 1.0 s termina circa **30.8%** dei campioni;
- entro 1.5 s termina circa **99.4%**.

Quindi quasi tutto l'input reale è terminato prima dell'ultimo mezzo secondo della finestra.

PLIF è forte proprio nella regione in cui gli eventi sono presenti o sono terminati da poco.

---

# 43. Diagnostica event-aligned

Con readout valutato rispetto all'ultimo evento, PLIF migliora rispetto a B:

| Offset dall'ultimo evento | PLIF − B Macro-F1 |
|---|---:|
| L | +7.66 pp |
| +50 ms | +9.03 pp |
| +100 ms | +9.80 pp |
| +200 ms | +9.95 pp |
| +300 ms | +9.95 pp |
| +400 ms | +8.56 pp |
| +500 ms | +7.23 pp |
| +600 ms | +5.04 pp |
| +700 ms | +2.15 pp |
| +800 ms | +0.54 pp |
| +950 ms | −0.24 pp |

Questo è un fenomeno molto chiaro:

> **PLIF conserva uno stato discriminativo forte per alcune centinaia di millisecondi dopo la fine degli eventi.**

---

# 44. Stato profondo persistente

Attività del LIF nello stage2-attention dopo l'ultimo evento:

| Offset | B | PLIF |
|---|---:|---:|
| L | .5117 | .6172 |
| +100 ms | .4130 | .5182 |
| +200 ms | .2546 | .3729 |
| +300 ms | .1120 | .2336 |
| +400 ms | .0399 | .1291 |
| +500 ms | .0215 | .0642 |
| +800 ms | .0171 | .0081 |

PLIF mantiene dunque più a lungo attività interna semanticamente utile.

Le tau apprese mostrano anche una forte differenziazione:

- `stage2.attn.attn_lif`: media \(\tau \approx 6.227\), range circa 4.820–7.086;
- `q_lif`: \(\tau \approx 1.442\);
- `proj_lif`: \(\tau \approx 1.672\).

La rete sembra quindi apprendere una memoria lunga nel punto più profondo dell'attenzione.

---

# 45. Perché PLIF viene comunque scartato

Tra 1.5 e 2.0 secondi:

- B cresce di circa **+13.72 pp F1**;
- PLIF cresce soltanto di circa **+4.93 pp**.

Il problema non è un improvviso “late collapse”, ma una crescita insufficiente nella parte finale.

Inoltre:

- nessun cutoff `last_event + K` supera B alla metrica finale;
- `sum/t` e `sum/40` danno dinamiche molto simili;
- quindi non basta cambiare normalizzazione del readout.

Conclusione:

> PLIF ha scoperto una dinamica interessante, ma **non è un candidato architetturale finale** nella forma testata.

Il suo valore è diagnostico: dimostra che la scelta della dinamica neuronale può modificare radicalmente **quando** l'informazione è disponibile, anche se la metrica finale non migliora.

---

# PARTE VII — VARIANTI DI GATING TEMPORALE

## 46. Gated v2 e famiglia diagonale

È stata esplorata una famiglia di temporal gating diagonale / tanh.

La variante principale testata raggiunge:

- parametri: circa **501,476**
- Accuracy: **19.80%**
- Macro-F1: **17.60%**

e fallisce il gate di overfit.

Questa famiglia viene quindi chiusa.

La lezione metodologica è importante:

> un gate temporale apparentemente leggero non è necessariamente più facile da ottimizzare o più adatto di un mixer causale residuale.

TCAP beneficia di una proprietà decisiva: **zero-init e identity path esatto**.

---

# PARTE VIII — RAPPRESENTAZIONI TEMPORALI FINI: TBR

## 47. Perché esplorare TBR

Il binning E0 usa 50 ms per timestep.

Questo riduce drasticamente la risoluzione nativa dell'event stream.

L'ipotesi è che alcune informazioni siano perse già **prima** del Transformer.

Per testare la questione si implementa una codifica TBR con:

- 8 microbin per ogni macrobin;
- durata microbin: **6.25 ms**;
- 40 macro timestep totali.

In questo modo si preserva la lunghezza \(T=40\) del backbone, ma ogni timestep contiene una descrizione interna della struttura temporale più fine.

---

# 48. Limite semantico della TBR canonica

La formulazione TBR pubblicata utilizzata come riferimento è **polarity-agnostic**.

Questo significa che:

- mantiene occupancy temporale fine;
- perde la distinzione ON/OFF;
- perde molteplicità di eventi multipli nello stesso microbin.

Questa è una modifica semantica sostanziale rispetto a E0.

---

# 49. F+TBR

Parametri:

> **431,004**

Il modello riesce a overfittare quasi completamente:

- train accuracy ≈ 100%;
- validation accuracy nel gate ≈ 98.44%;

ma non porta la validation loss sotto il threshold richiesto:

> min val loss ≈ **1.5368 > 1.5**.

Quindi fallisce il gate.

La conclusione corretta è:

> la rappresentazione è apprendibile, ma nella forma testata non dà un segnale sufficientemente pulito/stabile per meritare un full sweep.

Non viene aperta una ricerca iperparametrica su TBR, per evitare esplosione combinatoria.

---

# 50. Spike-TBR-LIF

Viene inoltre implementata una variante più aderente alla formulazione spike-TBR con neurone:

- \(\beta = 0.9\);
- threshold = **1.1**.

Un'analisi diagnostica mostra però un forte collo di bottiglia di firing nell'encoder: soltanto circa **9.82%** dei voxel canonical-TBR risulta non-zero dopo la dinamica spiking nella configurazione analizzata.

Una diversa semantica di reset aumenterebbe molto l'attività, ma non viene scelta arbitrariamente perché il reset nella fonte non è sufficientemente specificato per giustificare un cambio.

Questo è un altro esempio di rigore metodologico:

> non viene modificata la semantica neuronale solo perché produce più spike, se la modifica non è supportata dalla formulazione di riferimento.

---

# PARTE IX — RAPPRESENTAZIONE MULTIGRANULARE

## 51. Motivazione

Le prove TBR suggeriscono che aumentare semplicemente la risoluzione della codifica può introdurre nuovi problemi.

La strategia cambia quindi.

Invece di sostituire il percorso coarse, si conserva integralmente E0 e si aggiunge un **secondo percorso fine**.

L'idea è:

> **il percorso a 50 ms mantiene stabilità, polarità e conteggi; un branch ad alta frequenza aggiunge micro-dinamiche locali.**

Questo evita di obbligare il modello a scegliere tra:

- rappresentazione coarse robusta;
- rappresentazione fine potenzialmente fragile.

---

# 52. MultiGranular-Lite

Input coarse:

```text
40 × 2 × 128 × 128
```

Fine stream:

```text
320 × 2 × 16 × 16
```

Il branch fine usa:

```text
Conv 3×3: 2 → 16
Conv 1×1: 16 → 64
continuous LIF
causal depthwise Conv1d
kernel/stride = 8
→ 40 output temporal steps
```

Le feature fine vengono sommate alla rappresentazione F:

```text
64 × 16 × 16
```

prima dello stage1.

Vantaggi:

- il Transformer continua a vedere **T=40**;
- polarità preservata;
- molteplicità preservata;
- nessun uso dell'endpoint reale del campione;
- overhead parametrico minimo.

Parametri:

> **433,188**

cioè soltanto:

> **+2,112 parametri rispetto a F.**

Il branch rappresenta il limite Pareto “lite” della famiglia.

---

# 53. MultiGranular-Capacity

Per testare se il concetto multigranulare ha bisogno di più capacità, viene progettata una variante più ricca:

- fine grid: **320 × 2 × 32 × 32**;
- learned residual downsample;
- proiezione `16 → 32 → 64`;
- temporal mixing MIMO `64 → 64` su gruppi da 8 timestep;
- residual fusion concat-conv.

Parametri:

> **480,036**

Quindi è ancora:

> **più piccola della baseline B.**

---

# 54. Risultato MG-Cap

F+MG-Cap ottiene:

- Accuracy: **48.95%**
- Macro-F1: **48.42%**
- Acc@1s: **41.28%**
- Acc@2s: **56.61%**
- Prefix-AUC: **0.2534**
- peak GPU: circa **4.94 GiB**
- tempo: circa **20.30 h**

Rispetto a F:

- +2.07 pp Accuracy circa;
- **+2.27 pp Macro-F1**.

Rispetto a B:

- **+4.14 pp Accuracy**
- **+4.27 pp Macro-F1**

Quindi il principio multigranulare è **positivo**.

Ma non è Pareto-dominant rispetto a F+TCAP.

---

# 55. Dove guadagna MG-Cap

La documentazione evidenzia che il guadagno è maggiormente concentrato nella prima parte temporale:

- Acc@1s: circa **+3.54 pp** rispetto a F;
- Acc@2s: circa **+0.60 pp** rispetto a F.

Questo è coerente con il ruolo intuitivo del branch fine:

> catturare micro-dinamiche locali e rendere più informativa la parte iniziale/intermedia della sequenza.

---

# 56. Costo di MG-Cap

La variante è molto più pesante in runtime:

> **20.30 h**

contro circa 6.6–6.7 h di F e F+TCAP.

Aumentano inoltre:

- stati temporali;
- compute del fine branch;
- memoria GPU.

Quindi MG-Cap viene considerata una prova positiva della famiglia, ma non ancora il candidato principale.

---

# 57. Clock-matched multigranular: un risultato negativo utile

È stato testato il tentativo di rendere le dinamiche LIF del branch fine “equivalenti” al clock coarse.

L'idea apparentemente naturale era conservare la stessa costante temporale fisica quando si passa da 50 ms a 6.25 ms.

Ma le varianti clock-matched falliscono il gate.

L'analisi porta a una conclusione importante:

> **uguaglianza della costante di decadimento non implica invarianza del blocco Conv-BN-LIF al cambio di clock.**

In particolare, l'input viene integrato 8 volte più frequentemente e il guadagno effettivo può crescere drasticamente.

Il codice viene quindi riportato alla versione stabile.

Questa è una lezione architetturale generale per le SNN multi-rate: la dinamica neuronale non può essere riscalata considerando soltanto \(\tau\); bisogna considerare l'intero sistema di integrazione, normalizzazione, threshold e firing.

---

# PARTE X — COMBINAZIONE MG-CAP + TCAP

## 58. Perché testare una sola combinazione

MG-Cap è positiva.

TCAP è la modifica temporale più forte.

La domanda naturale è se la fine temporal representation e il causal temporal mixer siano complementari.

Per evitare ricerca combinatoria, viene autorizzato **un solo test principale**:

> **MG-Cap + TCAP**

Configurazione:

```text
configs/dvslip_f_multigranular_temporal_capacity.yaml
```

Parametri:

> **541,476**

Lo stato attuale del branch `developer` registra questo modello come **final multigranular TCAP candidate**.

Il criterio di decisione stabilito è:

> se il guadagno rispetto a F+TCAP non è almeno nell'ordine di **+2 pp Macro-F1**, la famiglia multigranulare non viene mantenuta nella struttura finale.

Questo gate è importante perché protegge il progetto da una crescita incontrollata di complessità.

---

# PARTE XI — RAPPRESENTAZIONI SOSPESE O DEFERRED

## 59. E1 phase-count

E1 divide il conteggio di ciascun bin in due componenti legate alla fase temporale interna.

La proprietà interessante è:

> la somma delle componenti ricostruisce esattamente E0.

Quindi E1 aggiunge un primo momento temporale senza eliminare l'informazione coarse.

Costo stimato su B:

- parametri ≈ **501,284**
- incremento minimo dei parametri;
- ma circa **+377.49M multi-value MAC**.

La direzione viene sospesa per priorità sperimentale, non confutata.

---

# 60. Altre direzioni deliberatamente non aperte

Sono state considerate ma non portate a full experimental sweep:

- EST;
- HATS;
- TORE / TAF;
- Matrix-LSTM;
- MSTP completo;
- PSN / PMSN;
- GRU/LSTM-style temporal units;
- LMU;
- Mamba/state-space models;
- MTGA/event graph approaches;
- altre rappresentazioni ad alta frequenza.

La motivazione non è che siano idee sbagliate.

Il problema è di **attribuzione scientifica e budget sperimentale**.

Ogni nuova famiglia introduce:

- nuova semantica;
- nuovi iperparametri;
- nuovi controlli;
- rischio di confondere il motivo del guadagno.

La roadmap attuale preferisce chiudere prima una struttura semplice e interpretabile.

---

# PARTE XII — RISULTATI RIASSUNTIVI DVS-LIP

## 61. Tabella principale

| Variante | Parametri | Accuracy | Macro-F1 | Acc@1s | Acc@2s | Prefix-AUC | Peak GPU | Tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B** | 500,708 | 44.81 | 44.15 | 37.07 | 52.54 | .2453 | 4.58 GiB | 13.09 h |
| Capacity ~1M | 1,113,508 | 49.58 | 49.38 | — | — | .2932 | 6.79 GiB | 15.85 h |
| Capacity ~2M | 1,967,972 | 51.79 | 51.81 | — | — | .3365 | 9.00 GiB | 15.18 h |
| NoCrossTime | 500,708 | 15.13 | 13.01 | — | — | .0677 | — | — |
| mean@last_event | ~B | 42.50 | 42.02 | — | — | .3036 | — | — |
| last@last_event | ~B | 32.32 | 31.83 | — | — | .2171 | — | — |
| gated temporal | 501,476 | 19.80 | 17.60 | — | — | .0550 | — | — |
| **F** | **431,076** | **46.88** | **46.15** | 37.74 | 56.01 | .2366 | 3.41 GiB | 6.69 h |
| **B+TCAP** | 562,148 | 48.38 | 48.12 | 40.95 | 55.81 | .2611 | 4.64 GiB | 10.34 h |
| **B+PLIF** | 502,676 | 43.74 | 43.68 | 37.14 | 50.33 | **.2758** | 5.62 GiB | 13.38 h |
| **F+TCAP** | **492,516** | **52.79** | **52.36** | **44.42** | **61.15** | **.2916** | **3.50 GiB** | **6.61 h** |
| **F+MG-Cap** | 480,036 | 48.95 | 48.42 | 41.28 | 56.61 | .2534 | 4.94 GiB | 20.30 h |

---

# 62. Cosa emerge dalla tabella

## 62.1 Aumentare i parametri funziona, ma non è la soluzione desiderata

Capacity-2M migliora di:

> **+7.66 pp Macro-F1**

ma usa quasi quattro volte i parametri della baseline.

F+TCAP supera quel risultato restando sotto 500K.

---

## 62.2 La memoria temporale è necessaria

NoCrossTime crolla a:

> **13.01% Macro-F1**

quindi la dimensione temporale è realmente usata.

---

## 62.3 Il solo readout non basta

Le varianti event-aligned peggiorano.

Il limite è interno alla rappresentazione/dinamica, non unicamente nel modo in cui i logit vengono aggregati.

---

## 62.4 F è il miglior miglioramento hardware-oriented

F:

- migliora accuracy;
- riduce parametri;
- dimezza circa il footprint di stato;
- riduce il compute.

È quindi la base naturale del candidato finale.

---

## 62.5 TCAP è il miglior miglioramento temporale

TCAP produce un guadagno sostanziale e replicabile sul backbone.

La sua combinazione con F dà il miglior modello attuale.

---

## 62.6 PLIF insegna più di quanto dica la metrica finale

PLIF non va mantenuto, ma dimostra che:

- la memoria temporale profonda può essere allungata;
- la metrica finale può nascondere vantaggi temporali intermedi;
- gli stati post-evento possono essere semanticamente informativi.

---

## 62.7 MG-Cap è promettente ma costosa

MultiGranular-Capacity migliora F, soprattutto a tempo precoce, ma:

- richiede più stato;
- richiede più compute;
- aumenta fortemente il runtime.

La sua sopravvivenza dipende dal test combinato con TCAP.

---

# PARTE XIII — ANALISI HARDWARE

## 63. Perché il profiler hardware fa parte già della fase architetturale

L'obiettivo FPGA rende pericoloso valutare le architetture soltanto tramite parametri e accuracy.

Per una SNN bisogna distinguere almeno:

- firing rate;
- numero di stateful elements;
- state reads/writes;
- potential synaptic operations;
- activity-based synaptic operations;
- attention-specific operations;
- multi-value MAC;
- stima energetica teorica.

Il profiler v4 usa un campione fisso di **64 esempi validation round-robin** per confronti consistenti.

---

# 64. B vs F vs TCAP vs PLIF

| Metrica | B | F | B+TCAP | B+PLIF |
|---|---:|---:|---:|---:|
| firing rate | .0585 | .1352 | .0468 | .0525 |
| stateful elems | 1,099,776 | 477,184 | 1,198,080 | 1,099,776 |
| state reads | 42.89M | 18.61M | 45.84M | ~B |
| state writes | 45.09M | 19.56M | 46.17M | ~B |
| activity AC | 292.07M | 485.92M | 198.14M | 241.79M |
| potential AC | 5,557.45M | 2,904.56M | 5,557.45M | ~B |
| attention SOP | 11.14M | 11.14M | ~same | ~same |
| total potential SOP | 5,568.59M | 2,915.70M | ~B | ~B |
| multivalue MAC | 1,530.93M | 1,247.82M | 1,782.59M | ~B |

---

# 65. Interpretazione FPGA-oriented

### F

È il risultato più favorevole:

- meno memoria di stato;
- meno update;
- meno compute denso;
- frontend più compatto.

### TCAP

Richiede attenzione nel co-design:

- migliora temporal modeling;
- riduce spike activity;
- ma aggiunge dense channel mixing multi-value.

In un FPGA, TCAP potrebbe essere implementato in modo efficiente grazie a:

- ritardi fissi e piccoli;
- buffer circolare di `max_delay=4`;
- parallelizzazione sui canali;
- possibile quantizzazione dei \(W_\delta\);
- eventuale pruning/structured sparsity;
- time-multiplexing dei MAC.

Ma questa è una fase successiva: la struttura deve prima essere congelata.

### PLIF

Ha costo computazionale simile a B ma richiede tau apprendibili e aggiornamento del potenziale con parametri per canale.

Il guadagno finale insufficiente non giustifica l'aumento di complessità implementativa.

---

# 66. Limiti del proxy energetico

Le stime Horowitz utilizzate nel repository non sono misure reali su FPGA.

Il proxy non include completamente:

- accesso DRAM/BRAM;
- routing;
- leakage;
- comparatore LIF;
- reset;
- update membrana;
- pooling;
- classifier;
- costi di controllo;
- precisione effettiva futura.

Quindi i valori in µJ vanno letti come:

> **strumento relativo di confronto architetturale**, non come stima definitiva dell'energia del dispositivo.

---

# PARTE XIV — DVS-GESTURE COME CHECK SECONDARIO

## 67. Run attuale

Nel repository è presente anche un run DVS-Gesture:

- parametri: **489,227**
- Accuracy: **84.47%**
- Macro-F1: **83.62%**
- loss: **0.9882**

Profilo hardware:

- firing rate ≈ **0.02738**
- potential SOP ≈ **13,921.48M**
- multivalue MAC ≈ **3,827.30M**
- stateful elements: **1,099,776**
- Horowitz activity ≈ **17,949.51 µJ**
- dense ≈ **30,145.89 µJ**

Questi numeri **non devono essere confrontati direttamente** con DVS-Lip, perché cambiano:

- input geometry;
- protocollo;
- task;
- numero/durata dei timestep.

Il loro valore è soprattutto quello di verificare la portabilità generale della codebase e della famiglia di modelli.

---

# PARTE XV — PRINCIPI METODOLOGICI EMERSI

## 68. Non confondere performance con comprensione

Il mechanistic audit ha dimostrato che una singola accuracy non basta per capire il comportamento temporale.

La fase DVS-Lip mantiene questa lezione usando:

- Acc@1s;
- Acc@2s;
- Prefix-AUC;
- curve temporali;
- analisi event-aligned;
- firing/state diagnostics.

---

# 69. Non confondere più parametri con architettura migliore

I modelli 1M/2M costituiscono un controllo fondamentale.

Ogni modifica piccola deve dimostrare di essere migliore di un semplice capacity scaling in rapporto a:

- parametro;
- costo;
- stato;
- memoria;
- runtime.

F+TCAP soddisfa molto bene questo criterio.

---

# 70. Non aprire tuning prima di congelare la struttura

Il training recipe viene mantenuto il più possibile costante:

- optimizer;
- LR schedule;
- clipping;
- augmentation;
- loss;
- binning principale;
- seed di selezione.

Questo evita di attribuire al modello un miglioramento dovuto in realtà a hyperparameter tuning.

Il tuning avanzato è deliberatamente rimandato.

---

# 71. Usare gate economici prima dei full run

Le configurazioni più rischiose passano prima da bounded overfit / candidate gate.

Questo ha già evitato full run inutili per:

- gated-v2;
- TBR non sufficientemente stabile;
- clock-matched multigranular.

Il principio è:

> **spendere compute soltanto su configurazioni che hanno dimostrato almeno di poter rappresentare il training problem in modo numericamente sano.**

---

# 72. Un risultato negativo può chiudere una famiglia

La ricerca non è organizzata per “far funzionare ogni idea”.

Esempi:

- gated family: chiusa;
- PLIF: non final candidate;
- last-event readout: chiuso;
- clock-match naive: chiuso;
- TBR canonical: nessuno sweep;
- representation families più complesse: sospese.

Questo controllo del search space è essenziale dato il costo dei full run.

---

# PARTE XVI — STATO SCIENTIFICO ATTUALE

## 73. Cosa possiamo già sostenere

### Claim 1 — La temporalità è essenziale

Supporto:

- DVSGC temporal reversal/shuffle;
- NoCrossTime su DVS-Lip.

### Claim 2 — La baseline compatta non è semplicemente capacity-limited

Supporto:

- capacity scaling migliora;
- ma F+TCAP eguaglia/supera capacity-2M con circa un quarto dei parametri.

### Claim 3 — Una migliore geometria del frontend produce un Pareto gain

Supporto:

- F migliora Macro-F1;
- riduce parametri;
- riduce stato;
- riduce potential SOP;
- riduce proxy energetico.

### Claim 4 — Il temporal cross-channel mixing è utile

Supporto:

- B+TCAP +3.97 pp Macro-F1;
- F+TCAP +8.21 pp rispetto a B;
- guadagno non spiegabile dal solo numero di parametri.

### Claim 5 — Dinamiche neuronali apprendibili modificano fortemente il profilo temporale

Supporto:

- PLIF domina B a 1–1.5 s;
- tau profonde diventano molto più lunghe;
- persistenza event-aligned.

Ma non possiamo sostenere che PLIF migliori il modello finale.

### Claim 6 — L'alta frequenza locale contiene segnale utile, ma non è ancora una scelta Pareto

Supporto:

- MG-Cap > F;
- guadagno precoce;
- costo elevato.

---

# 74. Cosa NON possiamo ancora sostenere

Non è ancora corretto affermare che:

- F+TCAP sia definitivamente migliore in generalizzazione su DVS-Lip test;
- il miglioramento sia robusto a molti seed;
- il modello sia già ottimizzato per FPGA;
- la stima Horowitz corrisponda a energia reale;
- TCAP sia la migliore possibile formulazione temporale;
- il fine branch multigranulare sia necessario;
- le conclusioni del mechanistic audit DVSGC trasferiscano automaticamente a ogni dataset DVS;
- l'architettura finale sia già congelata.

Questi punti richiedono gli step successivi.

---

# PARTE XVII — PROSSIMI PASSI GIÀ DEFINITI

## 75. Chiudere MG-Cap+TCAP

Il primo passo è completare il candidate gate/full evaluation della configurazione:

```text
dvslip_f_multigranular_temporal_capacity.yaml
```

Decisione:

- se il guadagno su F+TCAP è chiaramente significativo e nell'ordine del gate prefissato, mantenere il branch multigranulare;
- altrimenti congelare **F+TCAP** come architettura strutturale.

---

# 76. Massimo un ulteriore probe locale ad alta frequenza

La roadmap consente al massimo un esperimento mirato aggiuntivo.

Lo scopo non è aprire una nuova famiglia, ma verificare un'ipotesi molto locale prima del freeze.

Questo limite è importante per impedire che la tesi diventi una ricerca infinita.

---

# 77. Replicazione multi-seed del confronto finale

Dopo il freeze architetturale:

- B;
- candidato finale;

vanno addestrati con **due nuovi seed comuni**, oltre all'attuale seed 42.

Da riportare:

- media ± std Accuracy;
- media ± std Macro-F1;
- Acc@1s;
- Acc@2s;
- Prefix-AUC;
- paired correctness;
- profiler hardware.

Questa è la fase che trasforma l'evidenza esplorativa attuale in un confronto molto più solido.

---

# 78. Soltanto dopo: augmentation e tuning

Una volta congelata l'architettura, si potrà aprire:

- event-specific augmentation;
- LR tuning;
- weight decay;
- regularizzazione;
- surrogate-gradient tuning;
- threshold/tau refinement;
- eventualmente distillation.

Questi miglioramenti dovranno essere presentati come **refinement**, non come contributo architetturale primario.

---

# 79. Fase hardware futura

Dopo il freeze:

- quantizzazione;
- pruning;
- structured sparsity;
- profiling su precisioni realistiche;
- mappatura buffer LIF;
- BRAM footprint;
- bandwidth;
- parallelismo;
- schedulazione del TCAP;
- eventuale RTL/HLS prototype.

Il lavoro già svolto con F rende questa fase più promettente perché il footprint di stato è stato ridotto drasticamente **prima** della quantizzazione.

---

# PARTE XVIII — NARRATIVA UNITARIA PER LA TESI

## 80. Il percorso non è una sequenza casuale di ablation

Il valore scientifico del lavoro sta soprattutto nella catena logica.

### Passo A — capire se la temporalità è realmente usata

DVSGC mostra che:

- il contenuto sopravvive;
- l'ordine viene distrutto da shuffle/reversal;
- la rete è quindi sensibile all'ordine.

### Passo B — rendere la diagnosi rigorosa

Si passa da perturbazioni singole a:

- matched inverse pairs;
- multi-seed;
- factorization;
- prefix;
- late utility;
- probe;
- causal patching;
- shortcut controls.

Una prima ipotesi di late harm viene anche confutata.

### Passo C — passare a un benchmark più realistico e difficile

DVS-Lip diventa il banco di sviluppo architetturale.

### Passo D — controllare se serve semplicemente più capacità

Capacity scaling dice sì, più capacità aiuta, ma a costo di modelli molto più grandi.

### Passo E — verificare la necessità della memoria

NoCrossTime collassa.

Quindi la temporalità non può essere eliminata.

### Passo F — ridurre il costo strutturale

Il frontend F migliora sia performance sia efficienza.

### Passo G — aumentare in modo mirato la capacità temporale

TCAP aggiunge cross-channel delayed mixing causale.

### Passo H — verificare complementarità

F+TCAP supera baseline e capacity-2M con ~492K parametri.

### Passo I — esplorare dinamiche e frequenza senza perdere controllo

PLIF e multigranular mostrano fenomeni utili, ma vengono giudicati anche per costo e stabilità.

### Passo J — congelare e validare

Il progetto è ora vicino al punto in cui smettere di esplorare e iniziare a consolidare.

---

# 81. Possibile formulazione del contributo principale

Una formulazione coerente con l'evidenza attuale potrebbe essere:

> Il lavoro mostra che una Mini-QKFormer fortemente compressa può essere migliorata in modo sostanziale su event streams temporali non attraverso un semplice aumento di capacità, ma tramite un co-design architetturale che combina una rappresentazione spaziale gerarchica più efficiente con mixing temporale causale cross-channel. L'analisi meccanicistica precedente permette inoltre di interpretare tali modifiche nel contesto della rappresentazione temporale effettivamente appresa dalla baseline, mentre il profiling hardware evidenzia come la geometria degli stati e il costo multi-value debbano essere considerati insieme alla firing sparsity.

Questa formulazione va aggiornata dopo:

- risultato MG-Cap+TCAP;
- multi-seed finale;
- test ufficiale;
- fase hardware.

---

# PARTE XIX — MAPPA DEI FILE IMPORTANTI

## 82. Mechanistic audit

```text
docs/mechanistic_temporal_audit.md
configs/mechanistic_audit_dvsgc_order2.yaml

src/etsr/data/matched_dvsgc.py
src/etsr/data/perturbations.py

src/etsr/evaluation/tracing.py
src/etsr/evaluation/metrics.py
src/etsr/evaluation/input_statistics.py
src/etsr/evaluation/probes.py
src/etsr/evaluation/causal.py
src/etsr/evaluation/mechanistic.py

scripts/prepare_matched_dvsgc.sh
scripts/train_audit_seed.sh
scripts/run_mechanistic_audit.sh

notebooks/archive/dvsgc/mechanistic_audit_analysis.ipynb
```

---

# 83. Developer / DVS-Lip

Documentazione:

```text
docs/PROJECT_CHARTER.md
docs/PROJECT_STATUS.md
docs/DECISIONS.md
docs/EXPERIMENT_LEDGER.md
docs/ROADMAP.md
```

Modello:

```text
src/etsr/models/mini_qkformer.py
src/etsr/models/layers.py
src/etsr/models/spiking.py
src/etsr/models/temporal.py
```

Configurazioni principali:

```text
configs/dvslip_e0.yaml
configs/dvslip_f.yaml
configs/dvslip_b_temporal_capacity.yaml
configs/dvslip_f_temporal_capacity.yaml
configs/dvslip_b_plif.yaml
configs/dvslip_f_tbr.yaml
configs/dvslip_f_spike_tbr_lif.yaml
configs/dvslip_f_multigranular_lite.yaml
configs/dvslip_f_multigranular_capacity.yaml
configs/dvslip_f_multigranular_temporal_capacity.yaml
```

---

# PARTE XX — CHECKLIST DELLE EVIDENZE PER LA VERSIONE FINALE DELLA TESI

Prima di trasformare questo recap in materiale definitivo da tesi, sarà necessario aggiornare i seguenti punti.

- [ ] risultato definitivo MG-Cap+TCAP;
- [ ] scelta architettura frozen;
- [ ] multi-seed B vs final;
- [ ] intervalli di confidenza/paired statistics finali;
- [ ] test ufficiale DVS-Lip una sola volta nel protocollo finale;
- [ ] replica su almeno un secondo benchmark;
- [ ] augmentation/tuning separati dalle innovazioni strutturali;
- [ ] quantizzazione/pruning;
- [ ] profiler con precisioni reali;
- [ ] analisi BRAM/state buffering;
- [ ] stima/implementazione FPGA;
- [ ] figure schematiche finali di B, F, TCAP e modello finale;
- [ ] curve temporali multi-seed;
- [ ] tabella accuracy/params/state/SOP;
- [ ] ablation compatta e selezionata, non dump di tutti gli esperimenti.

---

# 84. Messaggio finale del percorso finora

Il lavoro compiuto finora può essere sintetizzato in una trasformazione progressiva del problema.

All'inizio la domanda era:

> “La nostra Mini-QKFormer sa davvero usare il tempo?”

L'audit ha mostrato che la risposta non è binaria. La rete usa l'ordine, conserva informazione temporale e cambia radicalmente comportamento sotto inversione, ma il suo profilo temporale deve essere analizzato separando contenuto, ordine, disponibilità e uso causale.

La domanda successiva è diventata:

> “Possiamo usare meglio il tempo senza perdere il carattere compatto e hardware-friendly del modello?”

La fase DVS-Lip ha dato finora una risposta positiva.

Il frontend F mostra che una migliore geometria può simultaneamente:

- migliorare la classificazione;
- ridurre parametri;
- ridurre stato;
- ridurre compute.

TCAP mostra che la memoria implicita LIF può essere integrata con un meccanismo temporale causale cross-channel molto più espressivo.

La combinazione F+TCAP porta la Macro-F1 da:

\[
44.15\% \rightarrow 52.36\%
\]

cioè:

\[
+8.21 \text{ pp},
\]

pur passando da:

\[
500{,}708 \rightarrow 492{,}516
\]

parametri.

Questo, allo stato attuale, è il risultato che meglio rappresenta il contributo centrale della tesi:

> **non più capacità in senso generico, ma capacità architetturale meglio collocata nello spazio e nel tempo.**

Il lavoro non è ancora concluso. La priorità adesso non è espandere ulteriormente il search space, ma verificare se la componente multigranulare merita di entrare nel candidato finale, congelare la struttura e trasformare il miglioramento osservato in un risultato robusto multi-seed e successivamente hardware-aware.

---

## Nota metodologica sulle cifre riportate

Le metriche DVS-Lip riportate in questo documento provengono dalla fase di sviluppo e selezione architetturale descritta nel branch `developer`. In particolare, il checkpoint è selezionato sulla medesima dev validation usata per molte analisi successive. Di conseguenza i confronti sono validi come **evidenza comparativa di ricerca**, ma non vanno presentati come stima finale imparziale della generalizzazione.

Analogamente, i numeri energetici sono proxy teorici e non misure FPGA.

Questa distinzione dovrà essere mantenuta esplicitamente anche nella tesi finale.

---

## Riferimenti interni principali

Questo recap è stato costruito a partire da:

- `docs/mechanistic_temporal_audit.md` nel branch `refactor/mechanistic-temporal-audit`;
- `notebooks/archive/dvsgc/mechanistic_audit_analysis.ipynb`;
- `docs/PROJECT_CHARTER.md`;
- `docs/PROJECT_STATUS.md`;
- `docs/DECISIONS.md`;
- `docs/EXPERIMENT_LEDGER.md`;
- `docs/ROADMAP.md`;
- implementazioni in `src/etsr/models/`;
- configurazioni `configs/dvslip_*.yaml`.

Il documento è volutamente organizzato per **argomento e causalità scientifica**, non per data dei commit.
