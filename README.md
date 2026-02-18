
# Benchmark di tecniche di Cloud Removal su dati Sentinel-1 e Sentinel-2

Questa repository contiene il codice per il progetto finale del corso di [Multimedia](https://web.dmi.unict.it/corsi/lm-18/insegnamenti?seuid=9A90BBCE-99E3-4FB0-BF91-CCAAA5C51791).

[Clicca qui](https://github.com/Kespers/cloud-removal-benchmark/blob/main/doc/document.pdf) per consultare la relazione finale.

## Setup

### 1. Clone modello per inferenza
Clona il repository ufficiale di DSen2-CR (necessario per l'architettura della rete):
```bash
git clone https://github.com/srl-tud/dsen2-cr.git dsen2-cr-main
```

### 2. Download del Dataset
Utilizza il notebook Jupyter fornito per scaricare il dataset di test (ROIs1868):
1. Apri `dataset_download.ipynb`.
2. Esegui le celle per scaricare ed estrarre i dati.
3. Verifica che la cartella `dataset/` contenga le sottocartelle `s1_asiaWest_test` e `s2_asiaWest_test` e il file `ground_truths.json`.

### 3. Setup dei Pesi del Modello
Scarica i pesi pre-addestrati e posizionali in `evaluation/checkpoints/`:
- File richiesto: `dsen2cr_carl_weights.hdf5`
- Percorso: `evaluation/checkpoints/dsen2cr_carl_weights.hdf5`

---

## Pipeline di Valutazione
Segui questi passaggi nell'ordine per riprodurre gli esperimenti

### A. Inferenza Batch (Docker + CPU)
Esegue il modello su tutte le patch specificate. Utilizza Docker per gestire le dipendenze legacy (TensorFlow 1.15) e multiprocessing per parallelizzare su CPU.

```bash
python evaluation/main.py --limit 150 --workers 4
```
*Questo genererà le predizioni nella cartella `evaluation/output/`.*

### B. Selezione del Miglior Candidato
Analizza le predizioni per ogni patch e seleziona la data con la minore copertura nuvolosa usando `s2cloudless`.

```bash
python evaluation/select_candidate.py
```
*Output: `evaluation/candidates.json`*

### C. Calcolo delle Metriche
Confronta il miglior candidato con la Ground Truth (immagine di riferimento senza nuvole) calcolando NRMSE, PSNR, SSIM e SAM.

```bash
python evaluation/metrics.py
```
*Output: Report dettagliato in `evaluation/metrics_report.csv`.*

### D. Visualizzazione dei Risultati
Lancia un'interfaccia grafica interattiva per confrontare visivamente le predizioni con la Ground Truth.

```bash
python evaluation/visualizer_app.py
```