import os
import numpy as np
import pandas as pd

class MusicDataset:
    def __init__(self, csv_path='data/sampled.csv', audio_dir='data/songs'):
        df = pd.read_csv(csv_path, sep='\t')

        # keep only required columns
        df = df[['track_id', 'lyrics', 'tag', 'language']]

        # rename for consistency
        df = df.rename(columns={'tag': 'genre'})

        # construct audio path
        df['audio_path'] = df['track_id'].apply(
            lambda x: os.path.join(audio_dir, f'{x}.mp3')
        )

        # reset index
        self.df = df.reset_index(drop=True)

    def get(self):
        return self.df
