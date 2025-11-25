import pandas as pd
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import os

CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'

def load_energy_hdf_to_pandas_2nd_house(h5_file_path, plot_data=True, use_time_axis=True, use_cyclic_encoding=True, 
                              weather_csv_path=None, print_debug=True):
    
    try:
        with pd.HDFStore(h5_file_path, mode='r') as store:
            keys = store.keys()
            if print_debug:
                print("HDFStore keys: " + str(keys))
    except Exception as e:
        if print_debug:
            print("Could not open HDFStore to list keys: " + str(e))

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
    
    col_electricity = 'E_total_cons_power'
    col_pv_power = 'E_prod_power'
    
    df_raw_nan_filled = fill_nan_values(df_raw, col_electricity, print_debug=print_debug)
    
    #as we are also exporting pv data for house E, we need to fill NaN values in E_pv_power as well
    df_raw_nan_filled = fill_nan_values(df_raw_nan_filled, col_pv_power, print_debug=print_debug)
    
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
    
    pv_forecast = df['E_prod_power']
    pv_forecast_smooth = pv_forecast.rolling(window=3, center=True, min_periods=1).mean()
    pv_forecast_smooth = pv_forecast_smooth.clip(lower=0.0)
    
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
    df_features['PV_forecast'] = pv_forecast_smooth.values
    
    #if I add the pv production, hopefully this does not distrups some access of the data where it was hardcoded with access -1 or so
    
    
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


def save_energy_df_as_mat(h5_file_path, out_path=None, plot_data=False, use_time_axis=True, use_cyclic_encoding=True, 
                              weather_csv_path=None, print_debug=True):
    """
    Load dataset via load_energy_hdf_to_pandas_2nd_house and save to a MATLAB .mat file.
    Neuer Parameter:
      - out_path: optionaler Pfad zur Ausgabedatei (.mat). Wenn None, wird h5_file_path + ".mat" verwendet.

    Gespeicherte Variablen:
      - time_datenum : MATLAB datenum (double)
      - time_str     : ISO timestamp strings (cell)
      - data         : numeric data array (n_samples x n_features) as double
      - col_names    : cell array mit Spaltennamen (Reihenfolge entspricht data)
    """
    
    df = load_energy_hdf_to_pandas_2nd_house(h5_file_path, plot_data=plot_data,
                                             use_time_axis=use_time_axis,
                                             use_cyclic_encoding=use_cyclic_encoding,
                                             weather_csv_path=weather_csv_path,
                                             print_debug=print_debug)
    if df is None:
        raise RuntimeError("Loaded dataframe is None; cannot save to .mat")

    # Index -> Datetime wenn möglich
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        idx = df.index


    def _datetimeindex_to_matlab_datenum(dt_index):
        py_dt = pd.to_datetime(dt_index).to_pydatetime()
        ordinals = np.fromiter((d.toordinal() for d in py_dt), dtype=np.int64)
        secs = np.fromiter(((d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6) for d in py_dt), dtype=np.float64)
        datenums = ordinals.astype(np.float64) + 366.0 + secs / 86400.0
        return datenums.astype(np.float64)

    time_datenum = _datetimeindex_to_matlab_datenum(idx)
    time_str = np.asarray(pd.to_datetime(idx).astype(str).tolist(), dtype=object)


    df_numeric = df.copy()
    col_names_all = list(df_numeric.columns)
    for c in col_names_all:
        df_numeric[c] = pd.to_numeric(df_numeric[c], errors='coerce')

    data_array = df_numeric.values.astype(np.float64)
    col_names = np.asarray(col_names_all, dtype=object)


    if out_path is None:
        base, _ = os.path.splitext(h5_file_path)
        out_path = base + ".mat"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mat_dict = {
        'time_datenum': time_datenum,
        'time_str': time_str,
        'data': data_array,
        'col_names': col_names
    }

    savemat(out_path, mat_dict, oned_as='row')

    if print_debug:
        print(GREEN + "Saved .mat file to: " + RESET + out_path)
        print("  Variables written: time_datenum (double), time_str (cell), data (double), col_names (cell)")
        print("  data shape: " + str(data_array.shape))

    return out_path


def get_peak_PV_production(df, n_peaks=10):
    """
    Calculate the peak PV production from the dataframe.
    Assumes the dataframe has a column 'PV_forecast' representing PV power in Watts.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'PV_forecast' column.
        n_peaks (int): Number of top peaks to return (max one per calendar day). Default: 10.
    Returns:
        tuple: (peaks, yearly_consumption)
            - peaks: list of dicts for each peak with keys: 'timestamp','date','value_w','value_kw','plot'.
            - yearly_consumption: list of dicts with keys: 'year', 'consumption_kwh'
    """
    # New behaviour:
    # - find top `n_peaks` peaks by PV_forecast (W)
    # - allow at most one peak per calendar day (when a peak is selected, other
    #   samples from the same day are not candidates)
    # - for each selected peak, plot the PV_forecast for the day +/- `days_window`
    # - return a list of dictionaries with timestamp, date, value in W and kW,
    #   and path to saved plot (if plotting succeeded)

    if 'PV_forecast' not in df.columns:
        raise ValueError("DataFrame does not contain 'PV_forecast' column.")

    # Parameters (tweakable)
    days_window = 3  # days before and after the peak day to plot
    save_plots = True
    out_dir = os.path.join("plots", "pv_top_peaks")

    # Ensure datetime index for easy slicing
    try:
        df_indexed = df.copy()
        df_indexed.index = pd.to_datetime(df_indexed.index)
    except Exception:
        df_indexed = df

    series = df_indexed['PV_forecast'].dropna()
    if series.empty:
        # still compute yearly consumption if possible
        yearly = []
        try:
            yearly = _compute_yearly_consumption(df_indexed)
        except Exception:
            yearly = []
        return ([], yearly)

    # Candidates sorted descending
    candidates = series.sort_values(ascending=False)

    selected = []
    used_days = set()

    for ts, val in candidates.items():
        day = pd.to_datetime(ts).date()
        if day in used_days:
            continue
        selected.append((pd.to_datetime(ts), float(val)))
        used_days.add(day)
        if len(selected) >= n_peaks:
            break

    if save_plots:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    for idx, (ts, val_w) in enumerate(selected, start=1):
        peak_kw = val_w / 1000.0
        peak_date = pd.to_datetime(ts).date()

        start = ts - pd.Timedelta(days=days_window)
        end = ts + pd.Timedelta(days=days_window)

        # Select window safely
        try:
            window = df_indexed.loc[start:end]['PV_forecast']
        except Exception:
            mask = (pd.to_datetime(df_indexed.index) >= start) & (pd.to_datetime(df_indexed.index) <= end)
            window = df_indexed.loc[mask, 'PV_forecast']

        plot_path = None
        if save_plots:
            try:
                fig, ax = plt.subplots(figsize=(12, 4))
                x = pd.to_datetime(window.index)
                y = window.values
                ax.plot(x, y, label='PV_forecast', color='tab:orange')
                # highlight the selected peak
                ax.axvline(ts, color='red', linestyle='--', alpha=0.7)
                ax.plot([ts], [val_w], 'ro', label='Peak')
                ax.set_title(f"Peak #{idx}: {peak_date} — {peak_kw:.2f} kW")
                ax.set_ylabel('PV_forecast (W)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()

                plot_path = os.path.join(out_dir, f"peak_{idx:02d}_{peak_date}.png")
                fig.savefig(plot_path)
                plt.close(fig)
            except Exception:
                plot_path = None

        results.append({
            'timestamp': ts,
            'date': peak_date,
            'value_w': val_w,
            'value_kw': peak_kw,
            'plot': plot_path,
        })

    # compute yearly consumption for full years
    try:
        yearly = _compute_yearly_consumption(df_indexed)
    except Exception:
        yearly = []

    return (results, yearly)


def _compute_yearly_consumption(df_indexed):
    """Compute total electricity consumption per year (kWh) for years where
    we have data from beginning to end of the year.

    Expects a DataFrame with a datetime-like index and a consumption column
    named 'Load' (Watts) or 'E_total_cons_power' as fallback.
    """
    # determine consumption column
    if 'Load' in df_indexed.columns:
        cons_col = 'Load'
    elif 'E_total_cons_power' in df_indexed.columns:
        cons_col = 'E_total_cons_power'
    else:
        raise ValueError("No load column found for yearly consumption (expected 'Load' or 'E_total_cons_power').")

    idx = pd.to_datetime(df_indexed.index)
    diffs = idx.to_series().diff().dropna()
    if len(diffs) == 0:
        return []
    dominant = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.median()
    dt_hours = float(dominant / pd.Timedelta(hours=1))

    years = sorted(pd.DatetimeIndex(idx).year.unique())
    results = []
    for year in years:
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)

        # check coverage
        if idx.min() <= year_start and idx.max() >= year_end:
            subset = df_indexed.loc[str(year)]
            # expected samples in year
            expected_samples = int(round(((year_end - year_start) / dominant))) + 1
            actual_samples = len(subset)
            # accept if at least 99% of expected samples are present
            if expected_samples > 0 and actual_samples >= 0.99 * expected_samples:
                # energy in kWh = sum(power_W) * dt_hours / 1000
                total_kwh = float(subset[cons_col].dropna().sum() * dt_hours / 1000.0)
                results.append({'year': int(year), 'consumption_kwh': total_kwh})
    return results

if __name__ == "__main__":
    h5_file_path = "data/data/dfE_300s.hdf"
    weather_csv_path = "data/data/weather_data_house_a_LUZ.csv"
    store_mat_path = "data/data/matlab/dfE_300s.mat"
    
    df = load_energy_hdf_to_pandas_2nd_house(h5_file_path, plot_data=False, use_time_axis=True, 
                                            use_cyclic_encoding=True, weather_csv_path=weather_csv_path,
                                            print_debug=True)
    # save_energy_df_as_mat(h5_file_path, out_path=store_mat_path, plot_data=False, use_time_axis=True, 
    #                       use_cyclic_encoding=True, weather_csv_path=weather_csv_path,
    #                       print_debug=True)
    
    peaks, yearly = get_peak_PV_production(df, 60)
    if not peaks:
        print("No PV peaks found.")
    else:
        print("Top PV peaks:")
        for i, p in enumerate(peaks, start=1):
            print(f"{i:2d}. {p['date']} — {p['value_kw']:.2f} kW (raw {p['value_w']:.0f} W)  plot: {p['plot']}")

    if yearly:
        print("\nYearly consumption for full years (kWh):")
        for y in yearly:
            print(f"{y['year']}: {y['consumption_kwh']:.1f} kWh")

    # Note: `plot_top_pv_events` is not defined in this module; if you want to
    # run additional event plotting, import or implement that function and
    # uncomment the call below.
    # results = plot_top_pv_events(df, n=5, window_days=2, use_time_axis=True, save_plots=True, out_dir="plots/pv_events_house_E")
    # print("Results of top PV events plotting:", results)
    
    