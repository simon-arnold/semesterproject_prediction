import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

# from rich.traceback import install
# install(show_locals=True)

# -----------------------------
# Dataset class
# -----------------------------
class EnergyDataset(Dataset):
    def __init__(self, df, seq_len=192, output_horizon=16, normalize=True, scaler=None, use_cyclic_encoding=True):
        """
        df: pandas DataFrame with columns:
            - Always present: Year, Month, Day, Timestep, Weekday, Load
            - If use_cyclic_encoding=True: tod_sin/cos, weekday_sin/cos, doy_sin/cos
        seq_len: length of the input sequence
        output_horizon: length of the output forecast (number of timesteps to predict)
        normalize: whether to normalize the data
        scaler: pre-fitted scaler (if None, will fit a new one - use for training data only!)
        use_cyclic_encoding: If True, uses cyclic sin/cos features. If False, uses raw time features.
        """
        self.seq_len = seq_len
        self.output_horizon = output_horizon
        self.use_cyclic_encoding = use_cyclic_encoding

        df = df.copy()
        
        # Store full dataframe for metadata access (used in plotting/evaluation)
        self._df_full = df.copy()

        # Define which columns go into the model based on encoding method
        # Check if Temperature column is available
        has_temperature = 'Temperature' in df.columns
        
        if use_cyclic_encoding:
            # Base: 8 features: Year + 6 cyclic (tod_sin/cos, weekday_sin/cos, doy_sin/cos) + Load
            self.model_feature_cols = ['Year', 'tod_sin', 'tod_cos', 'weekday_sin', 
                                        'weekday_cos', 'doy_sin', 'doy_cos']
            cols_to_normalize = ['Year']
            
            # Add Temperature as second-to-last feature if available
            if has_temperature:
                self.model_feature_cols.append('Temperature')
                cols_to_normalize.append('Temperature')
            
            # Load is always the last feature
            self.model_feature_cols.append('Load')
            cols_to_normalize.append('Load')
            
        else:
            # Base: 6 features: Year, Month, Day, Weekday, Timestep, Load
            self.model_feature_cols = ['Year', 'Month', 'Day', 'Weekday', 'Timestep']
            cols_to_normalize = ['Year', 'Month', 'Day', 'Weekday', 'Timestep']
            
            # Add Temperature as second-to-last feature if available
            if has_temperature:
                self.model_feature_cols.append('Temperature')
                cols_to_normalize.append('Temperature')
            
            # Load is always the last feature
            self.model_feature_cols.append('Load')
            cols_to_normalize.append('Load')
        
        # Check if required columns exist in DataFrame
        missing_cols = set(self.model_feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(RED +  f" Missing required columns in DataFrame: {missing_cols}\n"
                           f"   Available columns: {list(df.columns)}\n"
                           f"   Required columns: {self.model_feature_cols}" + RESET)
        
        # Optional normalization
        self.scaler = scaler
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
            # Use only model features (8 if cyclic, 6 if not)
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



        


    
    


