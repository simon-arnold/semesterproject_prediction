import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Dataset-Klasse
# -----------------------------
class EnergyDataset(Dataset):
    def __init__(self, df, seq_len=192, output_horizon=16, normalize=True):
        """
        df: pandas DataFrame mit Spalten ['Year','Month','Day','Timestep','Weekday','Load']
        seq_len: Länge der Input-Sequenz
        output_horizon: Länge des Outputs (z.B. 16 für 4x4)
        """
        self.seq_len = seq_len
        self.output_horizon = output_horizon

        # Optional normalisieren
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


# -----------------------------
# Funktion zum Laden der HDF5-Daten
# -----------------------------
def load_energy_df(h5_file_path):
    import pandas as pd

def load_energy_df(h5_file_path):
    """
    Lädt die HDF5-Datei und gibt einen DataFrame mit geglätteten 15-Minuten-Timesteps zurück.
    Berechnet 'Load' = A_total_cons_power - A_sauna_power.
    Jede 15-Minuten-Timestep ist der Mittelwert aus ±1 Step.
    Enthält nur die Spalten:
    Year, Month, Day, Timestep, Weekday, Load
    """
    df_raw = pd.read_hdf(h5_file_path, key='data')

    load = df_raw['A_total_cons_power'] - df_raw['A_sauna_power']

    load_smooth = load.rolling(window=3, center=True, min_periods=1).mean()
    # Ensure load cannot be negative (clip to 0)
    load_smooth = load_smooth.clip(lower=0)

    df_features = pd.DataFrame(index=df_raw.index)
    df_features['Year'] = df_raw.index.year
    df_features['Month'] = df_raw.index.month
    df_features['Day'] = df_raw.index.day
    df_features['Weekday'] = df_raw.index.weekday   # 0=Mon ... 6=Sun
    df_features['Timestep'] = (df_raw.index.hour * 60 + df_raw.index.minute) // 15 
    df_features['Load'] = load_smooth

    # --- 5. Keep only timestamps that correspond to the 4 timesteps per hour ---
    # i.e. only minutes 00, 15, 30, 45
    df_final = df_features[df_features.index.minute.isin([0, 15, 30, 45])]

    df_final = df_final.reset_index(drop=True)

    return df_final



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    file_path  = "data/data/dfA_300s.hdf"
    df = load_energy_df(file_path)
    # print(df.head())

    # df = pd.read_hdf(file_path, key='data')  # key='data' entspricht der Root-Gruppe

    print(len(df))
    print(df.head(50))
    print(df.columns)
    
    df['Load'].plot(figsize=(15,5), title='Load over time')
    plt.xlabel('Time')
    plt.ylabel('Load (kW)')
    plt.show()


        


    
    


