import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

YELLOW = '\033[93m'
RESET = '\033[0m'

# from rich.traceback import install
# install(show_locals=True)

# -----------------------------
# Dataset class
# -----------------------------
class EnergyDataset(Dataset):
    def __init__(self, df, seq_len=192, output_horizon=16, normalize=True, scaler=None):
        """
        df: pandas DataFrame with columns:
            - Metadata (stored but NOT used for model): 'Month', 'Day', 'Timestep', 'Weekday'
            - Model input features: 'Year', 'tod_sin', 'tod_cos', 'weekday_sin', 'weekday_cos', 
                                   'doy_sin', 'doy_cos', 'Load'
        seq_len: length of the input sequence
        output_horizon: length of the output (e.g. 16 for 4x4)
        normalize: whether to normalize the data (applies only to Year and Load)
        scaler: pre-fitted scaler (if None, will fit a new one - use for training data only!)
        """
        self.seq_len = seq_len
        self.output_horizon = output_horizon

        df = df.copy()
        
        # Store full dataframe for metadata access (used in plotting/evaluation)
        self._df_full = df.copy()

        # Define which columns go into the model (8 features total)
        # Order: Year, cyclic features (6), Load (last)
        self.model_feature_cols = ['Year', 'tod_sin', 'tod_cos', 'weekday_sin', 
                                    'weekday_cos', 'doy_sin', 'doy_cos', 'Load']
        
        # Check if required columns exist in DataFrame
        missing_cols = set(self.model_feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"❌ Missing required columns in DataFrame: {missing_cols}\n"
                           f"   Available columns: {list(df.columns)}\n"
                           f"   Required columns: {self.model_feature_cols}")
        
        # Optional normalization (only for Year and Load)
        # Cyclic features (tod_sin/cos, weekday_sin/cos, doy_sin/cos) are already in [-1,1]
        self.scaler = scaler
        cols_to_normalize = ['Year', 'Load']
        if normalize:
            if self.scaler is None:
                # Fit new scaler (only for training data!)
                self.scaler = MinMaxScaler()
                df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
            else:
                # Use pre-fitted scaler (for validation/test data)
                df[cols_to_normalize] = self.scaler.transform(df[cols_to_normalize])

        self.X, self.y = self.create_sequences(df)

    def create_sequences(self, df):
        X_list = []
        y_list = []
        input_start_date_list = []
        input_end_date_list = []
        
        for start_idx in range(0, len(df) - self.seq_len - self.output_horizon + 1):
            end_idx = start_idx + self.seq_len
            # Use only model features: Year + 6 cyclic + Load = 8 features
            seq_x = df.iloc[start_idx:end_idx][self.model_feature_cols].values
            seq_y = df.iloc[end_idx:end_idx+self.output_horizon]['Load'].values
            X_list.append(seq_x)
            y_list.append(seq_y)
            input_start_date_list.append(df.index[start_idx])
            input_end_date_list.append(df.index[end_idx-1])
           
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)

        self.input_start_dates = np.array(input_start_date_list, dtype='datetime64[ns]')
        self.input_end_dates = np.array(input_end_date_list, dtype='datetime64[ns]')

        
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



        


    
    


