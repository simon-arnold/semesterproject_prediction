import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'

def load_energy_hdf_to_pandas_2nd_house(h5_file_path, plot_data=True, use_time_axis=True, use_cyclic_encoding=True, 
                              weather_csv_path=None, print_debug=True):
    
    # try:
    #     with pd.HDFStore(h5_file_path, mode='r') as store:
    #         keys = store.keys()
    #         if print_debug:
    #             print("HDFStore keys: " + str(keys))
    # except Exception as e:
    #     if print_debug:
    #         print("Could not open HDFStore to list keys: " + str(e))

    try:
        df_raw = pd.read_hdf(h5_file_path, key='data')
    except (KeyError, ValueError):
        # Fallback: nimm ersten Key aus dem Store
        with pd.HDFStore(h5_file_path, mode='r') as store:
            store_keys = store.keys()
            if len(store_keys) == 0:
                raise RuntimeError("No keys found in HDF5 file: " + h5_file_path)
            first_key = store_keys[0]
            if print_debug:
                print("Key 'data' not found, falling back to first key: " + str(first_key))
            df_raw = store.get(first_key)

    # --- Assert timestep resolution is 5 minutes ---
    try:
        idx = pd.to_datetime(df_raw.index)
        diffs = idx.to_series().diff().dropna()
        if len(diffs) < 1:
            raise AssertionError("Index hat weniger als 2 Einträge; Timestep kann nicht bestimmt werden.")
        
        mode = diffs.mode()
        dominant = mode.iloc[0] if len(mode) > 0 else diffs.median()
        expected = pd.Timedelta(minutes=5)
        if dominant != expected:
            raise AssertionError("Erwarteter Timestep: 5 minutes, erkannt: {}. Prüfe Daten oder resample auf 5min.".format(dominant))
    except Exception as e:
        raise AssertionError("Konnte Timestep-Assertion nicht durchführen: " + str(e))


    if print_debug:
        print("Columns found: " + str(list(df_raw.columns)))
    
    
    if plot_data:
        try:
            fig1 = plot_dataset(df_raw, use_time_axis=use_time_axis)
           
            fig2 = plot_total_cons_power(df_raw, use_time_axis=use_time_axis, print_debug=print_debug)
           
            try:
                plt.show()
            except Exception:
                os.makedirs("plots", exist_ok=True)
                if fig1 is not None:
                    fig1.savefig(os.path.join("plots", "dataset_plot.png"))
                if fig2 is not None:
                    fig2.savefig(os.path.join("plots", "D_total_cons_power.png"))
                if print_debug:
                    print("No GUI display available — plots saved to: plots/")
        except Exception as e:
            if print_debug:
                print("Failed to plot dataset: " + str(e))
    
    # Here the conversion from the input df to the needed df starts:
    
    col = 'E_total_cons_power'
    
    df_raw_nan_filled = fill_nan_values(df_raw, col, print_debug=print_debug)
    
    df = convert_df_to_expected_format(df_raw_nan_filled, use_cyclic_encoding=use_cyclic_encoding,weather_csv_path=weather_csv_path, print_debug=print_debug)
        
    return df

def fill_nan_values(df, column_name, print_debug=True):
    
    """
    Fills NaN values in the specified column using forward-fill and backward-fill methods.
    Args:
        df: pandas DataFrame
        column_name: name of the column to fill NaNs in
    Returns:
        DataFrame with NaNs filled in the specified column
    """
    amount_nan_before = df[column_name].isna().sum()
    longest_nan_seq = df[column_name].isna().astype(int).groupby(df[column_name].notna().astype(int).cumsum()).sum().max()
    longest_nan_seq_time = longest_nan_seq * (df.index[1] - df.index[0])
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    df[column_name] = df[column_name].fillna(method='ffill').fillna(method='bfill')
    
    if print_debug:
        print("Filled " + str(amount_nan_before) + " NaN values. Longest NaN sequence was " + str(longest_nan_seq) + " entries. This is " + str(longest_nan_seq_time) + " time.")
    
    return df
    
def convert_df_to_expected_format(df, use_cyclic_encoding=True, weather_csv_path=None, print_debug=True):
    """
    Converts the input DataFrame to the expected format for the EnergyDataset house dataset.
    Finally will have the features: Month, Day, Timestep, Weekday, tod_sin/cos, weekday_sin/cos, doy_sin/cos, day_of_year, Year, Load, Temperature

    Args:
        df (pd.DataFrame): Pandas DataFrame with raw Load and Time features
        use_cyclic_encoding (bool, optional): Jusges if time of day, weekday, day of year gets coded using sin/cos encoding. Defaults to True.

    Raises:
        NotImplementedError: _description_
    """
    if not use_cyclic_encoding:
        raise NotImplementedError("Only use_cyclic_encoding=True is implemented for any other house than house A")
    
    t_timestep = 0.25 # [h]
    P_day = 24 / t_timestep
    P_week = 7
    P_year = 365
    
    df.index = pd.to_datetime(df.index)
    
    load = df['E_total_cons_power']
    load_smooth = load.rolling(window=3, center=True, min_periods=1).mean()
    load_smooth = load_smooth.clip(lower=0.0) 
    
    dt_index = pd.DatetimeIndex(df.index)
    df_features = pd.DataFrame(index=dt_index)
    
    df_features['Month'] = dt_index.month
    df_features['Day'] = dt_index.day
    df_features['Timestep'] = (dt_index.hour * 60 + dt_index.minute) // 15 
    df_features['Weekday'] = dt_index.weekday   # 0=Mon ... 6=Sun
    
    #encoding the cyclical features
    df_features['tod_sin'] = np.sin(2 * np.pi * df_features['Timestep'] / P_day)
    df_features['tod_cos'] = np.cos(2 * np.pi * df_features['Timestep'] / P_day)
    

    df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['Weekday'] / P_week)
    df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['Weekday'] / P_week)
    

    df_features['day_of_year'] = dt_index.dayofyear
    df_features['doy_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / P_year)
    df_features['doy_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / P_year)
    
    assert df_features['Timestep'].max() <= 95, "Timestep exceeds 95"
    assert df_features['Weekday'].max() <= 6, "Weekday exceeds 6"
    
    # Add Year and Load
    df_features['Year'] = dt_index.year
    df_features['Load'] = load_smooth.values
    
    # Now create the final Dataset which will be returned (still missing the temperature!)
    df_final = df_features[dt_index.minute.isin([0, 15, 30, 45])]
    df_final.index = pd.to_datetime(df_final.index.values)
    
    # Adding the weather!
    df_weather = load_weather_data(weather_csv_path, print_debug=print_debug)
    df_final = df_final.join(df_weather, how='inner')
    
    if print_debug:
            print("     Merged dataframe shape: " + str(df_final.shape))
            print("     Temperature column added: " + str(df_final['Temperature'].notna().sum()) + " valid values")
            print("     Temperature stats: mean=" + "{:.1f}".format(df_final['Temperature'].mean()) + "°C, "
                + "std=" + "{:.1f}".format(df_final['Temperature'].std()) + "°C, "
                + "min=" + "{:.1f}".format(df_final['Temperature'].min()) + "°C, "
                + "max=" + "{:.1f}".format(df_final['Temperature'].max()) + "°C\n")
            
            print("\nFinal Dataframe, first 5 rows:")
           
           
            with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
                print(df_final.head(5).to_string())
            
    return df_final
    
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
    
    df_weather = pd.read_csv(weather_csv_path, sep=';', decimal=',')
    
    df_weather['timestamp'] = pd.to_datetime(df_weather['reference_timestamp'], format='%d.%m.%Y %H:%M')
    df_weather = df_weather.set_index('timestamp')
    df_weather = df_weather.sort_index()
    
    if 'tre200s0' not in df_weather.columns:
        raise ValueError("Column 'tre200s0' not found in weather data. Available columns: " + str(df_weather.columns.tolist()))
    
    temperature = df_weather['tre200s0'].copy()
    temperature = pd.to_numeric(temperature, errors='coerce')

    
    amount_nan_before = temperature.isna().sum()
    longest_nan_seq = temperature.isna().astype(int).groupby(temperature.notna().astype(int).cumsum()).sum().max()
    longest_nan_seq_time = longest_nan_seq * (temperature.index[1] - temperature.index[0])
    temperature = temperature.ffill().bfill()
    
    print("Filled " + str(amount_nan_before) + " NaN values in temperature data. Longest NaN sequence was " + str(longest_nan_seq) + " entries. This is " + str(longest_nan_seq_time) + " time.")
    
    print(GREEN + "Weather data loaded: " + RESET + str(len(temperature)) + " records\n")
    
    if print_debug:
        print("     Date range: " + str(temperature.index[0]) + " to " + str(temperature.index[-1]))
        print("     Temperature range: " + "{:.1f}".format(temperature.min()) + "°C to " + "{:.1f}".format(temperature.max()) + "°C")
    
    
    temp_15min = pd.Series(dtype=float)
    idx_10min = temperature.index
    
    
    start_date = idx_10min[0].floor('h')  
    end_date = idx_10min[-1].ceil('h')   
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
    
    temp_15min = temp_15min.ffill().bfill()
    
    if print_debug:
        print("     Resampled to 15-minute intervals: " + str(len(temp_15min)) + " records")
        print("     NaN values after resampling: " + str(temp_15min.isna().sum()))
    
    return pd.DataFrame({'Temperature': temp_15min})


def plot_dataset(df, use_time_axis=True):
    """
    Plots each column of the DataFrame in its own subplot with a shared x-axis.
    Args:
        df: pandas DataFrame (index should be datetime-like for time axis)
        use_time_axis: if True, xlabel is 'Time' and index will be used as datetime axis if possible
    """
    if df is None or df.shape[1] == 0:
        raise ValueError("DataFrame is empty or None")

    cols = list(df.columns)
    n = len(cols)

    x = df.index
    if use_time_axis:
        try:
            x = pd.to_datetime(df.index)
        except Exception:
            
            x = df.index

    
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        y = df[col].values
        ax.plot(x, y, linewidth=0.9, alpha=0.9)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Optional: einfache Statistik in Ecke
        try:
            mean_val = np.nanmean(y)
            std_val = np.nanstd(y)
            stats = "μ={:.2f}, σ={:.2f}".format(mean_val, std_val)
            ax.text(0.98, 0.92, stats, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
        except Exception:
            pass

    # x-label nur unten
    xlabel = "Time" if use_time_axis else "Index"
    axes[-1].set_xlabel(xlabel, fontsize=11)

    plt.tight_layout()
    
    return fig

def plot_total_cons_power(df, use_time_axis=True, print_debug=True):
    """
    Plot a separate figure for the column 'D_total_cons_power' if it exists.
    If no DISPLAY is available, saves the plot to plots/D_total_cons_power.png.
    """
    col = 'E_total_cons_power'
    if col not in df.columns:
        if print_debug:
            print("Column '" + col + "' not found in dataframe. Skipping separate plot.")
        return

    x = df.index
    if use_time_axis:
        try:
            x = pd.to_datetime(df.index)
        except Exception:
            x = df.index

    y = df[col].values

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, linewidth=1.0, alpha=0.9, color='tab:blue')
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(col + " (separate plot)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Statistics (nan-aware)
    mean_val = np.nanmean(y)
    std_val = np.nanstd(y)
    nan_count = int(np.sum(np.isnan(y)))
    stats = "μ={:.2f}, σ={:.2f}, NaN={}".format(mean_val, std_val, nan_count)
    ax.text(0.98, 0.92, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    plt.tight_layout()
    # return figure to caller; caller will show/save
    return fig

if __name__ == "__main__":
    h5_file_path = "data/data/dfE_300s.hdf"
    weather_csv_path = "data/data/weather_data_house_a_LUZ.csv"
    
    df = load_energy_hdf_to_pandas_2nd_house(h5_file_path, plot_data=False, use_time_axis=True, 
                                            use_cyclic_encoding=True, weather_csv_path=weather_csv_path,
                                            print_debug=True)