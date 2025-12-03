import numpy as np
import torch
from torch.utils.data import Dataset


class NpzDataset(Dataset):
    def __init__(self, npz_path, seq_len=30):
        data = np.load(npz_path)
        print("LOADED DATA: ", data)
        # self.X = df[['open','high','low','close','volume']].values
        # self.y = df['action'].map({"long":0,"short":1,"stop":2,"no_action":3}).values
        self.X = data["X"]
        self.y = data["y"]
        print("loaded stock_sequences.npz", self.X.shape, self.y.shape)
        # loaded stock_sequences.npz (4828, 30, 8) (4828,)
        print(self.X[10])

        self.feature_mean = self.X.mean(axis=(0, 1))     # (8,)
        self.feature_std  = self.X.std(axis=(0, 1)) + 1e-8
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # print("get item & apply norm")
        seq = (self.X[idx] - self.feature_mean) / self.feature_std

        feats = torch.tensor(seq, dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        # print(feats)
        return feats, label
