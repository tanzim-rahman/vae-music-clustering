import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def extract_mfcc_features(tsv_path, audio_dir, n_mfcc=20, sr=22050):
    df = pd.read_csv(tsv_path, sep='\t')

    features, genres, languages = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        track_id = row['track_id']
        file_path = os.path.join(audio_dir, f'{track_id}.mp3')

        try:
            y, _ = librosa.load(file_path, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            feature = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1)
            ])

            features.append(feature)
            genres.append(row['tag'])
            languages.append(row['language'])

        except Exception as e:
            print(f'Error: {track_id} -> {e}')

    return np.array(features), np.array(genres), np.array(languages)
