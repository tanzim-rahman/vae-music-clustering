import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

def cluster_purity(true_labels, pred_labels):
    clusters = np.unique(pred_labels)
    total = len(true_labels)
    purity_sum = 0

    for cluster in clusters:
        # ignore noise from DBSCAN
        # not used in this project
        if cluster == -1:
            continue

        # data points in the cluster
        indices = np.where(pred_labels == cluster)[0]

        if len(indices) == 0:
            continue

        # true labels of data points
        true_cluster_labels = true_labels[indices]

        # true label frequencies
        counts = np.bincount(true_cluster_labels)

        # add dominant class to purity
        purity_sum += np.max(counts)

    # returns fraction of dominant classes
    return purity_sum / total

def evaluate_clustering(X, labels, true_labels):
    results = {}

    results["Silhouette"] = silhouette_score(X, labels)
    results["Calinski"] = calinski_harabasz_score(X, labels)
    results["Davies"] = davies_bouldin_score(X, labels)

    results["ARI"] = adjusted_rand_score(true_labels, labels)
    results["NMI"] = normalized_mutual_info_score(true_labels, labels)
    results["Purity"] = cluster_purity(true_labels, labels)

    return results
