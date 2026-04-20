from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

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
