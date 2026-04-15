import numpy as np
from sklearn.preprocessing import StandardScaler

def fuse(z_audio, X_lyrics, weight=0.5):
    z_audio = StandardScaler().fit_transform(z_audio)
    X_lyrics = StandardScaler().fit_transform(X_lyrics)
    return np.concatenate([z_audio, weight*X_lyrics], axis=1)
