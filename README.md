# music-genre-classifier

- Rivedere l'ultima trasformata che ha citato il professore (Mel-Frequency Cepstral Coefficients (MFCC))

- Suddivisione in cartelle dei vari generi musicali -> campionare dataset a random -> pre-processing file audio -> calcoliamo spettrogramma con tutti i modi differenti (separatamente) -> CNN -> training di classificazione fatto confrontando il dato classificato con la directory di riferimento

- Dubbi su pre-processing: quali tecniche di pre-processing utilizziamo?

	Non classifichiamo un intero brano musicale, utilizzare il finestramento del dato: spezzettare la traccia musicale in N finestre (durata non fissa, variabile sulla base del dato, di solito 2-5 sec, anche 6), tanti campioni ottenuti a partire da una singola traccia (ci sono molti esempi in rete), altrimenti avremmo una traccia che si estende in lunghezza con un'intera analisi tempo-frequenza. Consigliati 5 sec

	E' utile utilizzare finestre diverse?

	Il contenuto musicale non dipende dalla durata, quanto dal numero di battute (BPM), ci sono dei software online che identificano le battute per secondo del brano in modo tale da identificare la durata di una misura musicale, alcuni brani hanno ripetizione/cambio di tema ogni 4/8/16 battute, fattore di molteplicità legato alla numerosità delle battute, che ovviamente dipende del metronomo che adoperiamo e possono corrispondere a divere finestre temporali; bisogna ragionare più musicalmente che non fisicamente: rapportare lunghezza a numero di bit del secondo del metronomo

	A seconda della traccia prendiamo una finestra di dimensionalità diversa: scoprire se un brano è un walzer o un blues le lunghezze di misura sono diverse, gli accenti sono diversi, confrontarli col tempo non ha molto senso, bisogna confrontarli con un numero fisso di battute

	Non spendiamo troppe energie nella generazione del dataset: consigliano di trovare qualcosa in rete già organizzato per classi, noi soffermiamoci su processamento del dato: finestra -> unificazione -> singoli dimensionamenti rispetto alla traccia del genere; il dataset deve avere la stessa lunghezza per non avere bias, è importante non introdurli; train/test split e applichiamo k-fold cross validation (stratified k-fold o group k-fold) permette di utilizzare porzioni di dato musicale non solo nel training, ma anche nel test; utilizziamo label per brano per identificare il brano, un codice, ciò permette un buon processo di validazione del modello

	Finestre diverse per confrontare risultati in maniera globale, non in maniera indipendente

	Riduzione del rumore e filtraggio di alcune frequenze importanti? Aggiunta di parti di pre-processing aiuta per classificazione, potrebbe essere utile: vedere in letteratura lo stato dell'arte, quello che è stato fatto negli ultimi anni dai ricercatori, come guida, se iniziamo adesso a fare stato dell'arte, sicuramente prendiamo spunto e riprendiamo tecniche da utilizzare nell'articolo scientifico, SI DEVE PRENDERE SPUNTO PER TESTARE

- Quanti generi vogliamo rappresentare?
	
	Non abbiamo un'idea precisa, vorremmo fare un numero significativo, una decina, più o meno

	Bisogna adoperare un campione che sia coerente: se facciamo 4 generi (jazz modal, blues, walzer e pop) molto probabilmente da questi 4 generi bisogna estrarre i campioni; con un modello supervisionato bisogna estrarre campioni coerenti con le classi che cataloghiamo, altrimenti il problema diventa non supervisionato ed è più complicato, non lo abbiamo visto nel corso, sarebbe un problema di clusterizzazione. Il problema invece è supervisionato e dobbiamo 'dirgli' le classi a prescindere

- Trasformazioni tempo-frequenza, quali vogliamo utilizzare? Possiamo esplorare anche altro

	Wavelet + short time fourier trasform + fourier trasform le abbiamo approfondite in un altro corso e possiamo utilizzarle

- Dubbi esame: la discussione del progetto viene datta singolarmente oppure tutti insieme?

	E' importante ottenere massime performance di classificazione, verrà valutato la presentazione dei risultati, se fatta in modo chiaro o meno, una volta ottenuti i risultati bisogna esporre bene i risultati, esporli bene anche con visualizzazione grafica del dato

	Bisogna essere padroni dei processi di trasformazione + progettazione dell'architettura + presentazione dei risultati, ognuno esporrà la parte su cui ha maggior padronanza

- Non utilizzare una sola pipeline di pre-processing: tre sottosezioni in cui ognuno di noi si specializza sul sotto processamento del dato, e ognuno espone la parte che si sente più propria; non è una sfida a chi ha performance di classificazione migliore: bisogna fare analisi del dato musicale, iniziare finestramento su quello che sono i generi musicali, sconsigliato il MIDI perchè è una rappresentazione della musica e quindi il processamento del dato/l'analisi in frequenza non sarebbe opportuna

- Meglio utilizzare modello non pre-addestrato

- Commentiamo bene il codice che ci aiuta per la stesura del paper

- 23 Luglio ore 12 va bene come data di consegna per il progetto? 

	Sentiamoci qualche giorno prima per decidere consegna, possiamo contattare via mail per dubbi lato codice e analisi dei dati. Mandare articolo 1/2 gg prima della discussione

	Una presentazione come esposizione del progetto va bene

## SETUP

- Utilizzare PyTorch
- Utilizzare libreria "librosa" (contiene tutte le trasformate)
- Le immagini degli spettrogrammi del dataset https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification sono state generate tramite Short-time Fourier Transform 

## TODO

### Pre-processing

- [x] Divisione in battiti delle varie tracce (BPM & BPS)
- [ ] Valutare se serve finestramento
- [X] Applicazione delle trasformate (Short-time Fourier transform, Wavelet, Mel-Frequency Cepstral Coefficients (MFCC)) -> spettrogramma
- [ ] Processing degli spettrogrammi per migliorere qualità immagini (es. logaritmica)

### Training

- [X] Divisione dataset (train/test split)
- [ ] Scelta architettura
- [ ] Applicazione della CNN
- [ ] Training