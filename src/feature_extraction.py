import librosa
import numpy as np

def extract_mfcc(path, n_mfcc=40):
    y, sr = librosa.load(path)

    # trim silence
    y, _ = librosa.effects.trim(y)

    # ensure fixed length (30 seconds)
    y = librosa.util.fix_length(y, size=30*sr)

    # normalise audio (amplitude)
    y = librosa.util.normalize(y)

    # MFCC feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return np.mean(mfcc.T, axis=0)
