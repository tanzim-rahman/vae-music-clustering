from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import cm

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
    plt.savefig(f'results_easy/plots/{filename}', bbox_inches='tight')
    plt.show()
