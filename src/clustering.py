from sklearn.cluster import KMeans, AgglomerativeClustering

def kmeans(X, k=6):
    return KMeans(n_clusters=k, random_state=27).fit_predict(X)

def agglo(X, k=6):
    return AgglomerativeClustering(n_clusters=k).fit_predict(X)
