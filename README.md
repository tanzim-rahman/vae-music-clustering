# Unsupervised Music Genre Clustering using Variational Autoencoders

## Overview

This project implements an **unsupervised learning pipeline** using a **Variational Autoencoder (VAE)** to learn latent representations of music tracks and perform clustering. The goal is to group songs based on learned features from audio.

---

## Installation

```bash
git clone https://www.github.com/tanzim-rahman/vae-music-clustering

cd vae-music-clustering

pip install -r requirements.txt
```

---

## Usage

1. Download the **GTZAN** dataset available [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

2. Copy the *genres_original* directory from the dataset into the *data* directory of the project.

3. Open **train.ipynb** using Jupyter Notebook and run the cells.

---

## Dataset

We use the **GTZAN Dataset Music Genre Classification** dataset, which contains:

- 1000 audio tracks
- 10 music genres (blues, classical, country etc.)
- 30-second WAV files

---

## Preprocessing Steps

1. **Audio Validation**

   - Corrupted files are removed during dataset loading

2. **Audio Cleaning**

   - Silence trimming
   - Fixed duration (30 seconds)
   - Amplitude normalisation

3. **Feature Extraction**

   - MFCC (Mel-Frequency Cepstral Coefficients)

4. **Feature Normalisation**

   - StandardScaler applied across dataset

---

## Model Architecture

### Variational Autoencoder (VAE)

- Encoder: Fully connected layers (Input -> 256 -> 128)
- Latent space: 16-dimensional
- Decoder: Mirror image of encoder (128 -> 256 -> Output)
- Loss:

  - Reconstruction Loss (MSE)
  - KL Divergence

---

## Clustering Method

- K-Means

---

## Evaluation Metrics

- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalised Mutual Information (NMI)
- Cluster Purity

---

## Visualisations

- Training loss curves
- T-SNE plots of latent space

---

## Outputs

The following are saved in results/bvae_{BETA}:

- Training loss history (`.csv`)
- Training loss curve (`.svg`)
- Latent vectors (`.npy`)
- Cluster labels (`.npy`)
- True labels (`.npy`)
- Evaluation metrics (`.csv`)
- t-SNE visualisations (`.svg`)
- Model weights (`.pth`)

Additionally, the PCA results are saved in results/pca:

- PCA features (`.npy`)
- Cluster labels (`.npy`)
- True labels (`.npy`)
- Evaluation metrics (`.csv`)
- t-SNE visualisations (`.svg`)
