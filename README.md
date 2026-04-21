# Multimodal Music Clustering using VAE

This repository contains the implementation and evaluation of **unsupervised representation learning for music clustering** using deep learning models, developed as part of a Neural Networks course term project.

The project progresses across three levels of complexity:

- **Easy:** Audio-only Variational Autoencoder (VAE)
- **Medium:** Multimodal VAE (Audio + Lyrics)
- **Hard:** Multimodal Beta-VAE

## Repository Structure

```{txt}
data/
    songs/               # 30-second audio previews (.mp3)
    sampled.tsv          # Metadata (track_id, lyrics, genre, language etc.)

results_easy/
    plots/               # Loss curves, t-SNE plots
    ...                  # Trained VAE + evaluation metrics

results_medium/
    plots/               # Loss curves, t-SNE plots
    ...                  # Trained multimodal VAE + metrics

results_hard/
    plots/               # Loss curves, t-SNE, cluster distributions, reconstructions
    ...                  # Trained AE & Beta-VAE models + metrics

src/
    audio.py             # Audio feature extraction
    evaluation.py        # Clustering metrics
    models.py            # VAE, multimodal VAE, AE, Beta-VAE architectures
    visualisation.py     # Plotting utilities

easy.ipynb               # Easy task workflow
medium.ipynb             # Medium task workflow
hard.ipynb               # Hard task workflow

requirements.txt         # Dependencies
README.md                # You are here
```

## Dataset

The dataset was constructed by merging two sources:

- [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
- [Million Song Dataset + Spotify + Last.fm](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)

The former was used for the song metadata (lyrics/genre/language). The latter was used for the 30-second Spotify preview url.

Steps:

1. Matched songs by **title + artist**
2. Filtered to:
    - **500 English songs**
    - **500 German songs**
3. Downloaded audio previews

Important columns in the dataset:

- `track_id`
- `lyrics`
- `tag` (**Genre**: pop, rock, rap, country, rb, misc)
- `language` (en, de)

## Methodology

### Easy Task

- **Model:** Convolutional VAE (audio only)
- **Features:** MFCC
- **Baselines:** RAW, PCA
- **Clustering:** KMeans

### Medium Task

- **Model:** Multimodal Conv-VAE (audio + lyrics)
- **Text Features:** TF-IDF
- **Fusion:** Shared latent space
- **Baselines:** TEXT, RAW, PCA
- **Clustering:** KMeans, Agglomerative

### Hard Task

- **Model:** Multimodal Beta-VAE (Beta = 2, 4)
- **Baselines:**: TEXT, RAW, PCA, AE
- **Clustering:**: KMeans, Agglomerative

## Evaluation Metrics

### Easy

- Silhouette Score
- Calinski-Harabasz Index
- Adjusted Rand Index (ARI)

### Medium

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- ARI

### Hard

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- ARI
- Normalized Mutual Information (NMI)
- Cluster Purity

## Visualizations

Each `results_{task}/plots/` directory contains:

- **Loss Curves:** Training loss for all models
- **Latent Space (t-SNE):** Colored by Genre/Language
- **Cluster Distributions (Hard Task only):** Stacked bar plots (proportions) and heatmaps (cluster vs. label)

## How to Run

1. Clone repository.

    git clone <https://github.com/tanzim-rahman/vae-music-clustering>

    cd vae-music-clustering

2. Install dependencies.

    pip install -r requirements.txt

3. Run each notebook.

Please note that the *songs* directory is relatively large (approximately 360 MB), which may impact cloning time.
