import os
import re

import numpy as np
import pandas as pd

import librosa

from tqdm import tqdm

def clean_lyrics(text):
    text = text.lower()

    # remove [VERSE 1], [CHORUS] etc.
    text = re.sub(r'\[.*?\]', ' ', text)

    # remove (Verse 1), (Chorus) etc.
    text = re.sub(r'\(.*?\)', ' ', text)

    # remove newline and escape chars
    text = text.replace('\n', ' ').replace('\r', ' ')

    # remove non-alphabetic characters (keep German symbols)
    text = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', text)

    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

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
            print(f'Error: {track_id} - {e}')

    return np.array(features), np.array(genres), np.array(languages)

def extract_logmel(tsv_path, audio_dir, sr=22050, n_mels=64, duration=30):
    df = pd.read_csv(tsv_path, sep='\t')

    audio_features = []
    lyrics = []
    genres = []
    languages = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        track_id = row['track_id']
        file_path = os.path.join(audio_dir, f'{track_id}.mp3')

        try:
            y, _ = librosa.load(file_path, sr=sr, duration=duration)

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            logmel = librosa.power_to_db(mel)

            # Resize to fixed shape
            logmel = librosa.util.fix_length(logmel, size=128, axis=1)

            audio_features.append(logmel)
            lyrics.append(clean_lyrics(str(row['lyrics'])))
            genres.append(row['tag'])
            languages.append(row['language'])

        except Exception as e:
            print(f'Error: {track_id} - {e}')

    return np.array(audio_features), lyrics, np.array(genres), np.array(languages)
