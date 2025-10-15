import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def split_dataframe(df, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Chronological split of a dataframe into train/val/test sets.
    Ensures no overlap between splits.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    n = len(df)
    ntrain = int(n * train_frac)
    nval = int(n * val_frac)

    df_train = df.iloc[:ntrain]
    df_val   = df.iloc[ntrain:ntrain+nval]
    df_test  = df.iloc[ntrain+nval:]

    print(f"\nDataset split:")
    print(f"  Train: {len(df_train)} samples ({train_frac*100:.0f}%)")
    print(f"  Val:   {len(df_val)} samples ({val_frac*100:.0f}%)")
    print(f"  Test:  {len(df_test)} samples ({test_frac*100:.0f}%)")

    return df_train, df_val, df_test