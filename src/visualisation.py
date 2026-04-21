from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def plot_training_loss(loss_history, save_path=None):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

def plot_tsne(Z, labels, title, filename, cmap_name='tab10'):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    Z_2d = TSNE(n_components=2, random_state=27).fit_transform(Z)

    # Use chosen colormap
    cmap = cm.get_cmap(cmap_name, len(le.classes_))

    plt.figure(figsize=(8, 6))
    plt.scatter(
        Z_2d[:, 0],
        Z_2d[:, 1],
        c=labels_encoded,
        cmap=cmap,
        s=10
    )

    # Legend with matching colors
    handles = []
    for i, label in enumerate(le.classes_):
        handles.append(
            plt.Line2D([], [], marker='o', linestyle='',
                       color=cmap(i), label=label)
        )

    plt.legend(handles=handles, title='Classes',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_cluster_distribution(df, title, save_path=None, cmap='tab10'):
    # Count occurrences
    cluster_dist = pd.crosstab(df['cluster'], df['label'])

    # Normalise (proportions)
    cluster_dist_norm = cluster_dist.div(cluster_dist.sum(axis=1), axis=0)

    # Plot
    cluster_dist_norm.plot(
        kind='bar',
        stacked=True,
        figsize=(8,6),
        colormap=cmap
    )

    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    if save_path != None:
        plt.savefig(f'{save_path}.svg')
    plt.show()

    plt.figure(figsize=(8,6))
    sns.heatmap(
        cluster_dist_norm,
        annot=True,
        cmap='Blues',
        fmt='.2f',
        cbar=False
    )

    plt.title('Cluster vs Label Distribution (Heatmap)')
    plt.xlabel('Label')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(f'{save_path}_heatmap.svg')
    plt.show()

def plot_reconstruction(audio_orig, audio_recon, i, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    axs[0].imshow(audio_orig[i][0], aspect='auto', origin='lower')
    axs[0].set_title('Original')

    axs[1].imshow(audio_recon[i][0], aspect='auto', origin='lower')
    axs[1].set_title('Reconstructed')

    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()
