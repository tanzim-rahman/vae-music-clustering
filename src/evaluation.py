import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

def cluster_purity(y_true, y_pred):
    # Encode labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    total = 0
    N = len(y_true)

    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        true_labels = y_true[mask]

        if len(true_labels) == 0:
            continue

        # Count most frequent label in cluster
        counts = np.bincount(true_labels)
        total += np.max(counts)

    return total / N

def evaluation_easy(Z, true_labels, n_clusters):
    # Encode labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=27).fit(Z)
    y_pred = kmeans.labels_

    # Metrics
    sil = silhouette_score(Z, y_pred)
    ch = calinski_harabasz_score(Z, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    return kmeans, y_pred, sil, ch, ari

def evaluation_medium(Z, true_labels, n_clusters):
    # Encode labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=27).fit(Z)
    y_pred_kmeans = kmeans.labels_

    # Metrics
    sil_kmeans = silhouette_score(Z, y_pred_kmeans)
    ch_kmeans = calinski_harabasz_score(Z, y_pred_kmeans)
    d_kmeans = davies_bouldin_score(Z, y_pred_kmeans)
    ari_kmeans = adjusted_rand_score(y_true, y_pred_kmeans)

    # Agglo
    agglo = AgglomerativeClustering(n_clusters=n_clusters).fit(Z)
    y_pred_agglo = agglo.labels_

    # Metrics
    sil_agglo = silhouette_score(Z, y_pred_agglo)
    ch_agglo = calinski_harabasz_score(Z, y_pred_agglo)
    d_agglo = davies_bouldin_score(Z, y_pred_agglo)
    ari_agglo = adjusted_rand_score(y_true, y_pred_agglo)

    return (
        kmeans, y_pred_kmeans,
        agglo, y_pred_agglo,
        sil_kmeans, ch_kmeans, d_kmeans, ari_kmeans,
        sil_agglo, ch_agglo, d_agglo, ari_agglo,
    )

def evaluation_hard(Z, true_labels, n_clusters):
    # Encode labels to integers
    le = LabelEncoder()
    y_true = le.fit_transform(true_labels)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=27).fit(Z)
    y_pred_kmeans = kmeans.labels_

    # Metrics
    sil_kmeans = silhouette_score(Z, y_pred_kmeans)
    ch_kmeans = calinski_harabasz_score(Z, y_pred_kmeans)
    d_kmeans = davies_bouldin_score(Z, y_pred_kmeans)
    ari_kmeans = adjusted_rand_score(y_true, y_pred_kmeans)
    nmi_kmeans = normalized_mutual_info_score(y_true, y_pred_kmeans)
    cluster_kmeans = cluster_purity(y_true, y_pred_kmeans)

    # Agglo
    agglo = AgglomerativeClustering(n_clusters=n_clusters).fit(Z)
    y_pred_agglo = agglo.labels_

    # Metrics
    sil_agglo = silhouette_score(Z, y_pred_agglo)
    ch_agglo = calinski_harabasz_score(Z, y_pred_agglo)
    d_agglo = davies_bouldin_score(Z, y_pred_agglo)
    ari_agglo = adjusted_rand_score(y_true, y_pred_agglo)
    nmi_agglo = normalized_mutual_info_score(y_true, y_pred_agglo)
    cluster_agglo = cluster_purity(y_true, y_pred_agglo)

    return (
        kmeans, y_pred_kmeans,
        agglo, y_pred_agglo,
        sil_kmeans, ch_kmeans, d_kmeans, ari_kmeans, nmi_kmeans, cluster_kmeans,
        sil_agglo, ch_agglo, d_agglo, ari_agglo, nmi_agglo, cluster_agglo
    )
