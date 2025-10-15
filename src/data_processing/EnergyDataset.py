import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# from rich.traceback import install
# install(show_locals=True)

# -----------------------------
# Dataset class
# -----------------------------
class EnergyDataset(Dataset):
    def __init__(self, df, seq_len=192, output_horizon=16, normalize=True):
        """
        df: pandas DataFrame with columns ['Year','Month','Day','Timestep','Weekday','Load']
        seq_len: length of the input sequence
        output_horizon: length of the output (e.g. 16 for 4x4)
        """
        self.seq_len = seq_len
        self.output_horizon = output_horizon

        df = df.copy()

        # Optional normalization
        self.scaler = None
        if normalize:
            self.scaler = MinMaxScaler()
            df[['Year','Month','Day','Timestep','Weekday','Load']] = self.scaler.fit_transform(
                df[['Year','Month','Day','Timestep','Weekday','Load']])

        self.X, self.y = self.create_sequences(df)

    def create_sequences(self, df):
        X_list = []
        y_list = []
        for start_idx in range(0, len(df) - self.seq_len - self.output_horizon + 1):
            end_idx = start_idx + self.seq_len
            seq_x = df.iloc[start_idx:end_idx][['Year','Month','Day','Timestep','Weekday','Load']].values
            seq_y = df.iloc[end_idx:end_idx+self.output_horizon]['Load'].values
            X_list.append(seq_x)
            y_list.append(seq_y)
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def print_info(self, n_examples=1):
        print(f"Number of sequences: {len(self)}")
        print(f"Input shape: {self.X.shape}  | Output shape: {self.y.shape}")
        print(f"Input feature min/max: {self.X.min().item():.3f} / {self.X.max().item():.3f}")
        print(f"Output min/max: {self.y.min().item():.3f} / {self.y.max().item():.3f}")

        if n_examples > 0:
            import matplotlib.pyplot as plt
            for i in range(min(n_examples, len(self))):
                plt.figure(figsize=(12,4))
                plt.plot(self.X[i][:,-1].numpy(), label='Input Load sequence')
                plt.plot(range(self.X.shape[1], self.X.shape[1]+self.output_horizon), 
                         self.y[i].numpy(), 'r-o', label='Target horizon')
                plt.title(f"Sequence {i}")
                plt.xlabel("Timestep")
                plt.ylabel("Normalized Load")
                plt.legend()
                plt.show()



        


    
    


