import os
import torch
from torch.utils.data import Dataset
from src.feature_extraction import extract_mfcc
import librosa

class GTZANDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = [] # all valid files
        self.labels = [] # genre labels converted to numeric
        self.label_map = {} # mapping between genre string and int

        # list all genres and sort
        genres = sorted(os.listdir(root_dir))

        for i, genre in enumerate(genres):
            self.label_map[genre] = i
            genre_path = os.path.join(root_dir, genre)

            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    path = os.path.join(genre_path, file)

                    # validate file by checking if it loads successfully
                    try:
                        librosa.load(path, duration=1)
                        self.file_paths.append(path)
                        self.labels.append(i)
                    except:
                        print(f'Failed to load {path}')
                        continue

    def __len__(self):
        return len(self.file_paths)

    # returns the mfcc features and label of specific sound file
    def __getitem__(self, i):
        path = self.file_paths[i]
        label = self.labels[i]

        features = extract_mfcc(path)

        return torch.tensor(features, dtype=torch.float32), label
