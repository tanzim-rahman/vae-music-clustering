import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca(features, n_components=16):
    # normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # apply PCA
    pca = PCA(n_components=n_components, random_state=27)
    reduced = pca.fit_transform(features_scaled)

    return reduced, pca
