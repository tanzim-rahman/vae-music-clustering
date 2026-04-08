from sklearn.cluster import KMeans

def perform_kmeans(X, k=10):
    model = KMeans(n_clusters=k, random_state=27)
    return model.fit_predict(X)
