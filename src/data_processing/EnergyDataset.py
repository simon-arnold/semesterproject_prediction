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

def load_energy_hdf_to_pandas(h5_file_path, plot_data=True, use_time_axis=True):
    """
    Load the HDF5 file and return a DataFrame with smoothed 15-minute timesteps.
    Computes 'Load' = A_total_cons_power - A_sauna_power.
    Each 15-minute timestep is the mean of ±1 step (rolling window of 3).
    Returns only the columns:
    Year, Month, Day, Timestep, Weekday, Load
    """
    df_raw = pd.read_hdf(h5_file_path, key='data')

    # Check for NaN or zero values before any processing
    n_nan_raw = df_raw['A_total_cons_power'].isna().sum().sum()
    n_zero_raw = (df_raw == 0).sum().sum()
    print(f"Raw data check: {n_nan_raw} NaN values, {n_zero_raw} zeros in the raw dataset")

    # ensure datetime index (important for plotting time on the x-axis)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    
    # --- Fill NaNs (forward fill, then backward fill if needed) ---
    df_raw = df_raw.ffill().bfill()

    load = df_raw['A_total_cons_power'] - df_raw['A_sauna_power']

    load_smooth = load.rolling(window=3, center=True, min_periods=1).mean()
    # Ensure load cannot be negative (clip to 0)
    load_smooth = load_smooth.clip(lower=0)

    # explicit DatetimeIndex to avoid static analysis issues
    dt_index = pd.DatetimeIndex(df_raw.index)
    df_features = pd.DataFrame(index=dt_index)
    df_features['Year'] = dt_index.year
    df_features['Month'] = dt_index.month
    df_features['Day'] = dt_index.day
    df_features['Weekday'] = dt_index.weekday   # 0=Mon ... 6=Sun
    df_features['Timestep'] = (dt_index.hour * 60 + dt_index.minute) // 15 
    df_features['Load'] = load_smooth.values

    # --- Keep only timestamps that correspond to the 4 timesteps per hour ---
    # i.e. only minutes 00, 15, 30, 45
    df_final = df_features[dt_index.minute.isin([0, 15, 30, 45])]

    # ensure returned index is a clean DatetimeIndex without any serialized freq
    df_final.index = pd.to_datetime(df_final.index.values)
    
    print(len(df_final))
    print(df_final.head(5))
    print(df_final.columns)

    if plot_data:
        if use_time_axis:
            import matplotlib.dates as mdates
            
            x = pd.to_datetime(df_final.index).to_pydatetime()
            y = np.asarray(df_final['Load'].values)
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(x, y)
            ax.set_title('Load over time')
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            plt.xlabel('Time')
        else:
            df_num = df_final.reset_index(drop=True)
            ax = df_num['Load'].plot(figsize=(15, 5), title='Load over time (numeric index)')
            plt.xlabel('Index')

        plt.ylabel('Load (kW)')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
        
    n_nan_final = df_final.isna().sum().sum()
    n_zero_final = (df_final == 0).sum().sum()
    n_neg_final = (df_final < 0).sum().sum()
    print(f"Final data after cleaning check: {n_nan_final} NaN values, {n_zero_final} zeros, {n_neg_final} negative values in the final dataset")
    
    # Print rows with NaN values and their surrounding rows
    if n_nan_final > 0:
        print("\n--- Rows with NaN values (with context) ---")
        nan_mask = df_final.isna().any(axis=1)
        nan_indices = df_final[nan_mask].index
        
        for idx in nan_indices:
            idx_pos = df_final.index.get_loc(idx)
            # Get previous row (if exists)
            if idx_pos > 0:
                prev_idx = df_final.index[idx_pos - 1]
                print(f"\nPrevious row ({prev_idx}):")
                print(df_final.loc[prev_idx])
            
            # Current row with NaN
            print(f"\nRow with NaN ({idx}):")
            print(df_final.loc[idx])
            
            # Get next row (if exists)
            if idx_pos < len(df_final) - 1:
                next_idx = df_final.index[idx_pos + 1]
                print(f"\nNext row ({next_idx}):")
                print(df_final.loc[next_idx])
            
            print("-" * 50)

    return df_final

        


    
    


