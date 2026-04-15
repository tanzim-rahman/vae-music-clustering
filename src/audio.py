import numpy as np
import librosa

def extract_mfcc(file_path):
    # load audio
    y, sr = librosa.load(file_path, sr=22050)
    # extract mfcc features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    return mfcc[np.newaxis, :, :].astype(np.float32)

def flatten_mfcc(mfcc):
    return mfcc.reshape(-1)
