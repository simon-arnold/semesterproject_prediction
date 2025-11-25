import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'


def load_weather_data(weather_csv_path, print_debug=True):
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
    print(CYAN + "Loading weather data from: " + RESET + weather_csv_path + "\n")
    
    # Load CSV with proper parsing
    df_weather = pd.read_csv(weather_csv_path, sep=';', decimal=',')
    
    # Parse timestamp - format is "01.01.2010 00:00"
    df_weather['timestamp'] = pd.to_datetime(df_weather['reference_timestamp'], format='%d.%m.%Y %H:%M')
    df_weather = df_weather.set_index('timestamp')
    df_weather = df_weather.sort_index()
    
    # Extract temperature column (tre200s0)
    if 'tre200s0' not in df_weather.columns:
        raise ValueError("Column 'tre200s0' not found in weather data. Available columns: " + str(df_weather.columns.tolist()))
    
    temperature = df_weather['tre200s0'].copy()
    
    # Convert to numeric (in case there are string values)
    temperature = pd.to_numeric(temperature, errors='coerce')
    
    # Fill any NaN values
    temperature = temperature.ffill().bfill()
    
    print(GREEN + "Weather data loaded: " + RESET + str(len(temperature)) + " records\n")
    
    if print_debug:
        print("     Date range: " + str(temperature.index[0]) + " to " + str(temperature.index[-1]))
        print("     Temperature range: " + "{:.1f}".format(temperature.min()) + "°C to " + "{:.1f}".format(temperature.max()) + "°C")
    
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
    
    if print_debug:
        print("     Resampled to 15-minute intervals: " + str(len(temp_15min)) + " records")
        print("     NaN values after resampling: " + str(temp_15min.isna().sum()))
    
    return pd.DataFrame({'Temperature': temp_15min})


def load_energy_hdf_to_pandas(h5_file_path, plot_data=True, use_time_axis=True, use_cyclic_encoding=True, 
                              weather_csv_path=None, print_debug=True):
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
    
    if print_debug:
        print("     Raw data check: " + str(n_nan_raw) + " NaN values, " + str(n_zero_raw) + " zeros in the raw dataset")

    # ensure datetime index (important for plotting time on the x-axis)
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    
    # --- Fill NaNs (forward fill, then backward fill if needed) ---
    df_raw = df_raw.ffill().bfill()

    load = df_raw['A_total_cons_power'] - df_raw['A_sauna_power'] - df_raw['A_hp_power']

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
        df_weather = load_weather_data(weather_csv_path, print_debug=print_debug)
        
        # Merge on index (timestamps must match exactly)
        df_final = df_final.join(df_weather, how='inner')
        
        if print_debug:
            print("     Merged dataframe shape: " + str(df_final.shape))
            print("     Temperature column added: " + str(df_final['Temperature'].notna().sum()) + " valid values")
            print("     Temperature stats: mean=" + "{:.1f}".format(df_final['Temperature'].mean()) + "°C, "
                + "std=" + "{:.1f}".format(df_final['Temperature'].std()) + "°C, "
                + "min=" + "{:.1f}".format(df_final['Temperature'].min()) + "°C, "
                + "max=" + "{:.1f}".format(df_final['Temperature'].max()) + "°C\n")
    
    
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
    
    if print_debug:
        print("Final dataset info:")
        print("   Total records: " + str(len(df_final)))
        print("   Columns: " + str(list(df_final.columns)))
        print(df_final.head(5))
        print("Final data after cleaning check: " + str(n_nan_final) + " NaN values, " + str(n_zero_final) + " zeros, " + str(n_neg_final) + " negative values in the final dataset")
    
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
                print("\nPrevious row (" + str(prev_idx) + "):")
                print(df_final.loc[prev_idx])
            
            # Current row with NaN
            print("\nRow with NaN (" + str(idx) + "):")
            print(df_final.loc[idx])
            
            # Get next row (if exists)
            if idx_pos < len(df_final) - 1:
                next_idx = df_final.index[idx_pos + 1]
                print("\nNext row (" + str(next_idx) + "):")
                print(df_final.loc[next_idx])
            
            print("-" * 50)

    return df_final

def split_dataframe(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, print_debug=True):
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

    if print_debug:
        print(f"\nDataset split:")
        print(f"  Train: {len(df_train)} samples ({train_frac*100:.0f}%)")
        print(f"  Val:   {len(df_val)} samples ({val_frac*100:.0f}%)")
        print(f"  Test:  {len(df_test)} samples ({test_frac*100:.0f}%)")
    return df_train, df_val, df_test


if __name__ == "__main__":
    """
    Visualisiert die Features und Raw-Daten aus dem Energy-Dataset.
    
    Figure 1: Alle Features, die das Dataset rausgibt
    Figure 2: Alle einzelnen Verbrauchskomponenten aus df_raw
    """


    # Pfade
    h5_file = "data/data/dfA_300s.hdf"
    weather_csv = "data/data/weather_data_house_a_LUZ.csv"
    
    print("="*60)
    print("ENERGY DATASET VISUALIZATION")
    print("="*60)
    
    # Lade prozessierte Features (mit Temperature)
    df_features = load_energy_hdf_to_pandas(
        h5_file, 
        plot_data=False, 
        use_cyclic_encoding=True,
        weather_csv_path=weather_csv
    )
    
    # Lade Raw-Daten für Komponenten-Plot
    df_raw = pd.read_hdf(h5_file, key='data')
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()
    
    # Wähle Zeitbereich (z.B. 7 Tage für bessere Übersicht)
    start_date = df_features.index[0]
    end_date = start_date + pd.Timedelta(days=7)
    
    df_features_subset = df_features.loc[start_date:end_date]
    df_raw_subset = df_raw.loc[start_date:end_date]
    
    # Berechne zusätzliche Komponente: Total - Sauna (ohne Wärmepumpe)
    # Wichtig: Auf denselben Index wie df_features_subset reindexen (15-Minuten Daten)
    total_minus_sauna = (df_raw['A_total_cons_power'] - df_raw['A_sauna_power']).loc[start_date:end_date]
    total_minus_sauna_minus_hp = (df_raw['A_total_cons_power'] - df_raw['A_sauna_power'] - df_raw['A_hp_power']).loc[start_date:end_date]
    
    # Reindexiere auf 15-Minuten Index
    total_minus_sauna = total_minus_sauna.reindex(df_features_subset.index, method='nearest')
    total_minus_sauna_minus_hp = total_minus_sauna_minus_hp.reindex(df_features_subset.index, method='nearest')
    
    # =====================================================================
    # FIGURE 1: Alle Features die das Dataset rausgibt
    # =====================================================================
    print("\nCreating Figure 1: All Dataset Features...")
    
    fig1, axes1 = plt.subplots(4, 2, figsize=(16, 14))
    fig1.suptitle('Figure 1: Alle Features vom Energy-Dataset (7 Tage)', fontsize=16, fontweight='bold')
    
    # Load (ges - sauna - wärmepumpe)
    ax = axes1[0, 0]
    ax.plot(df_features_subset.index, total_minus_sauna, linewidth=1, label='Total - Sauna')
    ax.plot(df_features_subset.index, total_minus_sauna_minus_hp, linewidth=1, label='Total - Sauna - Wärmepumpe')
    ax.set_title('Load (Ges - Sauna - Wärmepumpe)', fontweight='bold')
    ax.set_ylabel('Load [W]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    

    
    # Temperature (falls vorhanden)
    ax = axes1[0, 1]
    if 'Temperature' in df_features_subset.columns:
        ax.plot(df_features_subset.index, df_features_subset['Temperature'], linewidth=1, color='red')
        ax.set_title('Temperature', fontweight='bold')
        ax.set_ylabel('Temperatur [°C]')
    else:
        ax.text(0.5, 0.5, 'Temperature nicht verfügbar', ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Cyclic: Time-of-Day (sin/cos)
    ax = axes1[1, 0]
    if 'tod_sin' in df_features_subset.columns:
        ax.plot(df_features_subset.index, df_features_subset['tod_sin'], label='tod_sin', linewidth=1, alpha=0.7)
        ax.plot(df_features_subset.index, df_features_subset['tod_cos'], label='tod_cos', linewidth=1, alpha=0.7)
        ax.set_title('Time-of-Day (cyclic)', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Cyclic Features nicht verfügbar', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Wert')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Cyclic: Weekday (sin/cos)
    ax = axes1[1, 1]
    if 'weekday_sin' in df_features_subset.columns:
        ax.plot(df_features_subset.index, df_features_subset['weekday_sin'], label='weekday_sin', linewidth=1, alpha=0.7)
        ax.plot(df_features_subset.index, df_features_subset['weekday_cos'], label='weekday_cos', linewidth=1, alpha=0.7)
        ax.set_title('Weekday (cyclic)', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Cyclic Features nicht verfügbar', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Wert')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Cyclic: Day-of-Year (sin/cos)
    ax = axes1[2, 0]
    if 'doy_sin' in df_features_subset.columns:
        ax.plot(df_features_subset.index, df_features_subset['doy_sin'], label='doy_sin', linewidth=1, alpha=0.7)
        ax.plot(df_features_subset.index, df_features_subset['doy_cos'], label='doy_cos', linewidth=1, alpha=0.7)
        ax.set_title('Day-of-Year (cyclic)', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Cyclic Features nicht verfügbar', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Wert')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Raw features: Month, Day, Weekday, Timestep
    ax = axes1[2, 1]
    ax.plot(df_features_subset.index, df_features_subset['Month'], label='Month', linewidth=1, alpha=0.7)
    ax.plot(df_features_subset.index, df_features_subset['Day'], label='Day', linewidth=1, alpha=0.7)
    ax.set_title('Month & Day', fontweight='bold')
    ax.set_ylabel('Wert')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax = axes1[3, 0]
    ax.plot(df_features_subset.index, df_features_subset['Weekday'], linewidth=1, color='purple')
    ax.set_title('Weekday (0=Mon, 6=Sun)', fontweight='bold')
    ax.set_ylabel('Wert')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax = axes1[3, 1]
    ax.plot(df_features_subset.index, df_features_subset['Timestep'], linewidth=1, color='orange')
    ax.set_title('Timestep (0-95, 15min intervals)', fontweight='bold')
    ax.set_ylabel('Wert')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Leer (für Symmetrie)
    
    
    plt.tight_layout()
    
    # =====================================================================
    # FIGURE 2: Alle einzelnen Verbrauchskomponenten aus df_raw
    # =====================================================================
    print("Creating Figure 2: Raw Power Components...")
    
    fig2, axes2 = plt.subplots(4, 2, figsize=(16, 12))
    fig2.suptitle('Figure 2: Alle Verbrauchskomponenten (Raw Data, 7 Tage)', fontsize=16, fontweight='bold')
    
    components = [
        ('A_total_cons_power', 'Gesamtverbrauch', 'blue'),
        ('A_dishwasher_power', 'Geschirrspüler', 'green'),
        ('A_stove_power', 'Herd', 'red'),
        ('A_exp_power', 'Export', 'orange'),
        ('A_hp_power', 'Wärmepumpe', 'purple'),
        ('A_sauna_power', 'Sauna', 'brown'),
        ('A_additional_power', 'Zusätzlich', 'cyan'),
        ('A_washing_machine_power', 'Waschmaschine', 'magenta')
    ]
    
    for idx, (col, title, color) in enumerate(components):
        ax = axes2[idx // 2, idx % 2]
        if col in df_raw_subset.columns:
            ax.plot(df_raw_subset.index, df_raw_subset[col], linewidth=1, color=color)
            ax.set_title(f'{title} ({col})', fontweight='bold')
            ax.set_ylabel('Power [W]')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, f'{col} nicht verfügbar', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    print("\nPlots erstellt! Fenster werden angezeigt...")
    print("   Figure 1: Alle Features vom Dataset")
    print("   Figure 2: Alle Raw-Verbrauchskomponenten")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Fertig!")
    print("="*60)