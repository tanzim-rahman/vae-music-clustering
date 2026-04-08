import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(latent, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=27)
    reduced = tsne.fit_transform(latent)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:,0], reduced[:,1], c=labels)
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

def plot_loss(loss_history, save_path):
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()
