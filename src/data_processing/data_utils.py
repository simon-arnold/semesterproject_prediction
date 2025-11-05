import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_weather_data(weather_csv_path):
    """
    Load weather data from CSV and resample to 15-minute intervals.
    
    The weather data is provided every 10 minutes. For 15-minute intervals:
    - xx:00 → use xx:00 (exact match)
    - xx:15 → average of xx:10 and xx:20
    - xx:30 → use xx:30 (exact match)
    - xx:45 → average of xx:40 and xx:50
    
    Args:
        weather_csv_path: Path to weather CSV file
        
    Returns:
        DataFrame with DatetimeIndex and 'Temperature' column at 15-minute intervals
    """
    print(f"\n📊 Loading weather data from: {weather_csv_path}")
    
    # Load CSV with proper parsing
    df_weather = pd.read_csv(weather_csv_path, sep=';', decimal=',')
    
    # Parse timestamp - format is "01.01.2010 00:00"
    df_weather['timestamp'] = pd.to_datetime(df_weather['reference_timestamp'], format='%d.%m.%Y %H:%M')
    df_weather = df_weather.set_index('timestamp')
    df_weather = df_weather.sort_index()
    
    # Extract temperature column (tre200s0)
    if 'tre200s0' not in df_weather.columns:
        raise ValueError(f"Column 'tre200s0' not found in weather data. Available columns: {df_weather.columns.tolist()}")
    
    temperature = df_weather['tre200s0'].copy()
    
    # Convert to numeric (in case there are string values)
    temperature = pd.to_numeric(temperature, errors='coerce')
    
    # Fill any NaN values
    temperature = temperature.ffill().bfill()
    
    print(f"   Weather data loaded: {len(temperature)} records")
    print(f"   Date range: {temperature.index[0]} to {temperature.index[-1]}")
    print(f"   Temperature range: {temperature.min():.1f}°C to {temperature.max():.1f}°C")
    
    # Create 15-minute resampled data
    temp_15min = pd.Series(dtype=float)
    
    # Get all timestamps in 10-minute resolution
    idx_10min = temperature.index
    
    # Generate target 15-minute timestamps
    start_date = idx_10min[0].floor('h')  # Start from beginning of first hour
    end_date = idx_10min[-1].ceil('h')    # End at end of last hour
    idx_15min = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    temp_15min_values = []
    
    for ts in idx_15min:
        minute = ts.minute
        
        if minute == 0:
            # xx:00 → use xx:00 (exact match)
            if ts in temperature.index:
                temp_15min_values.append(temperature.loc[ts])
            else:
                temp_15min_values.append(np.nan)
                
        elif minute == 15:
            # xx:15 → average of xx:10 and xx:20
            ts_10 = ts - pd.Timedelta(minutes=5)  # xx:10
            ts_20 = ts + pd.Timedelta(minutes=5)  # xx:20
            
            if ts_10 in temperature.index and ts_20 in temperature.index:
                temp_15min_values.append((temperature.loc[ts_10] + temperature.loc[ts_20]) / 2)
            else:
                temp_15min_values.append(np.nan)
                
        elif minute == 30:
            # xx:30 → use xx:30 (exact match)
            if ts in temperature.index:
                temp_15min_values.append(temperature.loc[ts])
            else:
                temp_15min_values.append(np.nan)
                
        elif minute == 45:
            # xx:45 → average of xx:40 and xx:50
            ts_40 = ts - pd.Timedelta(minutes=5)  # xx:40
            ts_50 = ts + pd.Timedelta(minutes=5)  # xx:50
            
            if ts_40 in temperature.index and ts_50 in temperature.index:
                temp_15min_values.append((temperature.loc[ts_40] + temperature.loc[ts_50]) / 2)
            else:
                temp_15min_values.append(np.nan)
    
    temp_15min = pd.Series(temp_15min_values, index=idx_15min, name='Temperature')
    
    # Fill any remaining NaN values
    temp_15min = temp_15min.ffill().bfill()
    
    print(f"   Resampled to 15-minute intervals: {len(temp_15min)} records")
    print(f"   NaN values after resampling: {temp_15min.isna().sum()}")
    
    return pd.DataFrame({'Temperature': temp_15min})


def load_energy_hdf_to_pandas(h5_file_path, plot_data=True, use_time_axis=True, use_cyclic_encoding=True, 
                              weather_csv_path=None):
    """
    Load the HDF5 file and return a DataFrame with smoothed 15-minute timesteps.
    Computes 'Load' = A_total_cons_power - A_sauna_power.
    Each 15-minute timestep is the mean of ±1 step (rolling window of 3).
    
    Args:
        h5_file_path: Path to the HDF5 file
        plot_data: Whether to plot the data
        use_time_axis: Whether to use time axis in plots
        use_cyclic_encoding: If True, adds cyclic sin/cos features (tod_sin/cos, weekday_sin/cos, doy_sin/cos)
                            If False, uses raw features (Month, Day, Weekday, Timestep)
        weather_csv_path: Optional path to weather CSV file. If provided, adds 'Temperature' column
    
    Returns:
        DataFrame with columns:
        - Always: Year, Month, Day, Timestep, Weekday, Load
        - If use_cyclic_encoding=True: also tod_sin/cos, weekday_sin/cos, doy_sin/cos
        - If weather_csv_path provided: also Temperature
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
    
    # Store base time features (always included)
    df_features['Month'] = dt_index.month
    df_features['Day'] = dt_index.day
    df_features['Timestep'] = (dt_index.hour * 60 + dt_index.minute) // 15 
    df_features['Weekday'] = dt_index.weekday   # 0=Mon ... 6=Sun
    
    # --- Conditionally add cyclic encodings ---
    if use_cyclic_encoding:
        # Time-of-day (Timestep 0-95, period = 96 = 24*4 timesteps per day)
        P_day = 96
        df_features['tod_sin'] = np.sin(2 * np.pi * df_features['Timestep'] / P_day)
        df_features['tod_cos'] = np.cos(2 * np.pi * df_features['Timestep'] / P_day)
        
        # Day-of-week (Weekday 0-6, period = 7)
        P_week = 7
        df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['Weekday'] / P_week)
        df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['Weekday'] / P_week)
        
        # Day-of-year (for seasonal patterns, period = 365)
        df_features['day_of_year'] = dt_index.dayofyear
        P_year = 365
        df_features['doy_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / P_year)
        df_features['doy_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / P_year)
    
    #Sanity check that Timestep, Weekday don't exceed their bounds
    assert df_features['Timestep'].max() <= 95, "Timestep exceeds 95"
    assert df_features['Weekday'].max() <= 6, "Weekday exceeds 6"
    
    # Add Year and Load
    df_features['Year'] = dt_index.year
    df_features['Load'] = load_smooth.values

    # --- Keep only timestamps that correspond to the 4 timesteps per hour ---
    # i.e. only minutes 00, 15, 30, 45
    df_final = df_features[dt_index.minute.isin([0, 15, 30, 45])]

    # ensure returned index is a clean DatetimeIndex without any serialized freq
    df_final.index = pd.to_datetime(df_final.index.values)
    
    # --- Merge weather data if provided ---
    if weather_csv_path is not None:
        print("\n🌡️  Integrating weather data...")
        df_weather = load_weather_data(weather_csv_path)
        
        # Merge on index (timestamps must match exactly)
        df_final = df_final.join(df_weather, how='inner')
        
        print(f"   Merged dataframe shape: {df_final.shape}")
        print(f"   Temperature column added: {df_final['Temperature'].notna().sum()} valid values")
        print(f"   Temperature stats: mean={df_final['Temperature'].mean():.1f}°C, "
              f"std={df_final['Temperature'].std():.1f}°C, "
              f"min={df_final['Temperature'].min():.1f}°C, "
              f"max={df_final['Temperature'].max():.1f}°C")
    
    print(f"\n📋 Final dataset info:")
    print(f"   Total records: {len(df_final)}")
    print(f"   Columns: {list(df_final.columns)}")
    print(df_final.head(5))

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