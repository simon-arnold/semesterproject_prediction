import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_processing.EnergyDataset import EnergyDataset
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.dates as mdates

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluates a trained model on the test set and computes metrics.

    Returns:
        metrics (dict): contains MSE, RMSE, MAE, MAPE
    """
    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss = criterion(y_pred, y_test)
            total_loss += loss.item() * X_test.size(0)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_test.cpu())

    total_loss /= len(test_loader.dataset)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(total_loss)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

    print("\n" + "=" * 50)
    print(" TEST SET PERFORMANCE")
    print("=" * 50)
    print(f"MSE:   {total_loss:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE:   {mae:.6f}")
    print(f"MAPE:  {mape:.2f}%")
    print("=" * 50 + "\n")

    return {"MSE": total_loss, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot_raw_dataframe(df, title="Test Data"):
    """
    Simply plots the raw DataFrame time series data (only 'Load' column).
    Can also plot multiple dataframes for train/val/test comparison.
    
    Args:
        df: pandas DataFrame with time series data, or tuple of (df_train, df_val, df_test)
        title: Plot title
    """

    if isinstance(df, tuple) and len(df) == 3:
        df_train, df_val, df_test = df

        print(f"{CYAN} Plotting train/val/test split...{RESET}")
        print(f"   Train: {len(df_train)} samples ({df_train.index[0]} to {df_train.index[-1]})")
        print(f"   Val:   {len(df_val)} samples ({df_val.index[0]} to {df_val.index[-1]})")
        print(f"   Test:  {len(df_test)} samples ({df_test.index[0]} to {df_test.index[-1]})")
        
        if 'Load' not in df_train.columns:
            print(f"Warning: 'Load' column not found. Using first column")
            load_col = df_train.columns[0]
        else:
            load_col = 'Load'

        # Calculate global Y-axis limits across all datasets
        global_min = min(df_train[load_col].min(), df_val[load_col].min(), df_test[load_col].min())
        global_max = max(df_train[load_col].max(), df_val[load_col].max(), df_test[load_col].max())
        
        # Add 5% padding to y-axis
        y_padding = (global_max - global_min) * 0.05
        y_min = global_min - y_padding
        y_max = global_max + y_padding
        
        # Calculate time spans for each dataset (in days)
        train_duration = (df_train.index[-1] - df_train.index[0]).total_seconds() / 86400
        val_duration = (df_val.index[-1] - df_val.index[0]).total_seconds() / 86400
        test_duration = (df_test.index[-1] - df_test.index[0]).total_seconds() / 86400
        max_duration = max(train_duration, val_duration, test_duration)
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=False)
        
        datasets = [
            (df_train, 'Train', 'blue'),
            (df_val, 'Validation', 'orange'),
            (df_test, 'Test', 'green')
        ]
        
        for i, (data, label, color) in enumerate(datasets):
            axes[i].plot(data.index, data[load_col].values, linewidth=0.8, color=color, alpha=0.8)
            axes[i].set_ylabel(f'{load_col} [W]', fontsize=11)
            axes[i].set_title(f"{label} Dataset ({len(data)} samples)", fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Set same Y-axis limits for all plots
            axes[i].set_ylim(y_min, y_max)
            
            # Set X-axis limits to ensure same visual spacing (days per inch)
            data_duration = (data.index[-1] - data.index[0]).total_seconds() / 86400
            
            # Calculate center of current dataset
            data_center = data.index[0] + (data.index[-1] - data.index[0]) / 2
            
            # Set x-limits to show the same time span as the longest dataset
            from datetime import timedelta
            x_min = data_center - timedelta(days=max_duration/2)
            x_max = data_center + timedelta(days=max_duration/2)
            axes[i].set_xlim(x_min, x_max)

            mean_val = data[load_col].mean()
            std_val = data[load_col].std()
            min_val = data[load_col].min()
            max_val = data[load_col].max()
            
            textstr = f'Mean: {mean_val:.1f} W | Std: {std_val:.1f} W | Min: {min_val:.1f} W | Max: {max_val:.1f} W'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes, fontsize=10,
                         verticalalignment='top', bbox=props)
        
        axes[-1].set_xlabel('Time', fontsize=12)
        fig.suptitle(f"{title} - Complete Dataset Split", fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1) 
        
    else:
        # Single dataframe
        print(f"Plotting raw dataframe...")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        # Check if 'Load' column exists
        if 'Load' not in df.columns:
            print(f"Warning: 'Load' column not found. Available columns: {list(df.columns)}")
            load_col = df.columns[0]  # Use first column as fallback
            print(f"   Using '{load_col}' instead")
        else:
            load_col = 'Load'
        
        # Create figure
        plt.figure(figsize=(20, 6))
        
        plt.plot(df.index, df[load_col].values, linewidth=0.8, color='blue', alpha=0.8)
        plt.ylabel(f'{load_col} [W]', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.title(f"{title} - {load_col} Time Series ({len(df)} samples)", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = df[load_col].mean()
        std_val = df[load_col].std()
        min_val = df[load_col].min()
        max_val = df[load_col].max()
        
        textstr = f'Mean: {mean_val:.1f} W\nStd: {std_val:.1f} W\nMin: {min_val:.1f} W\nMax: {max_val:.1f} W'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()

    print(f"{GREEN}Train/val/test plot generated!{RESET}")


def plot_multiple_predictions_at_date(model, test_set: EnergyDataset, start_date = None, 
                                      n_examples=1, seq_len=192, output_horizon=16, device="cpu", 
                                      plot_metrics=False):
    """
    Plot multiple predictions starting at a given date, spaced by the forecast horizon apart.
    
    Args:
        model: Trained model
        df_test: Test dataframe
        start_date: Last Date of the Input Sequence.
        n_examples: Number of predictions to plot (each spaced by output_horizon timesteps)
        seq_len: Sequence length for input
        output_horizon: Prediction horizon (number of timesteps to predict)
        scaler: MinMaxScaler for normalization
        device: Device for model inference
    """
    #TODO: The naming with start_date and pred_start_date does not exactly match what they actually represent.
    # The function itself works correctly. However, pred_start_time and its index should probably be shifted by 1 or
    # 15 minutes compared to the last date of the input (start_time). I handled this shift with the 15-minute offset in the print statement,
    # but not in the logic – this should be adjusted for completeness.
    # For the sake of completeness.

    print(f"{CYAN}Generating {n_examples} predictions from test set...{RESET}")
    
    # Generating a plot after each complete forecast horizon.
    spacing_timesteps = output_horizon
    spacing_hours = output_horizon / 4  # Each timestep = 15 minutes
    
    # Handle start_date
    if start_date is None:
        
        # Use earliest possible date (need seq_len history)
        actual_start = test_set.input_end_dates[0]
        closest_idx = 0
        print(f"   No start date provided. Using earliest possible date: {actual_start+pd.Timedelta(minutes=15)}")
        print(f"   Generating {n_examples} predictions spaced {spacing_hours:.1f} hours ({output_horizon} timesteps) apart...")
    else:
        print(f"   Generating {n_examples} predictions starting from {start_date}, spaced {spacing_hours:.1f} hours ({output_horizon} timesteps) apart...")

        if isinstance(start_date, str):
            start_date = np.datetime64(start_date) - np.timedelta64(15, 'm')
        else:
            raise NotImplementedError(
                "It is not clear what happens if the start_date input is not a string "
                "- this needs to be implemented and checked, especially regarding the 1 or 15 minute index/time offset."
            )

        if start_date not in test_set.input_end_dates:
            closest_idx = np.argmin(np.abs(test_set.input_end_dates - start_date))
            actual_start = test_set.input_end_dates[closest_idx]
            print(f"{RED}Exact date not found. Using closest date: {actual_start+np.timedelta64(15, 'm')}{RESET}")
        else:
            actual_start = start_date
            closest_idx = np.argmin(np.abs(test_set.input_end_dates - actual_start))

    
    for i in range(n_examples):
        pred_start_idx = closest_idx + i * spacing_timesteps
        
        input_data = test_set.X[pred_start_idx]
        target_data = test_set.y[pred_start_idx]
        pred_start_time = test_set.input_end_dates[pred_start_idx]
        
        print(f"   [{i+1}/{n_examples}] Prediction at {pred_start_time+np.timedelta64(15, 'm')}")
        
        
        X = input_data.unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
            pred_normalized = y_pred.cpu().numpy()
        
        # Denormalize predictions, inputs, and targets
        # Scaler fitted on different features depending on mode:
        # - Cyclic mode: ['Year', 'Load'] (2 features)
        # - Raw mode: ['Year', 'Month', 'Day', 'Weekday', 'Timestep', 'Load'] (6 features)
        # Load is always the LAST column in the scaler
        n_features = test_set.scaler.data_min_.shape[0]
        
        pred_full = np.zeros((len(pred_normalized[0]), n_features))
        pred_full[:, -1] = pred_normalized[0]  # Fill Load column
        pred_denormalized = test_set.scaler.inverse_transform(pred_full)[:, -1]
        
        input_full = np.zeros((len(test_set.X[pred_start_idx]), n_features))
        input_full[:, -1] = test_set.X[pred_start_idx][:, -1].cpu().numpy()  # Fill Load column
        input_load_denormalized = test_set.scaler.inverse_transform(input_full)[:, -1]
        
        target_full = np.zeros((len(test_set.y[pred_start_idx]), n_features))
        target_full[:, -1] = test_set.y[pred_start_idx].cpu().numpy()  # Fill Load column
        target_load_denormalized = test_set.scaler.inverse_transform(target_full)[:, -1]
               
        # Plot
        plt.figure(figsize=(16, 6))
        
        input_load = input_load_denormalized
        target_load = target_load_denormalized
        
        input_start_date = test_set.input_start_dates[pred_start_idx]
        input_end_date = test_set.input_end_dates[pred_start_idx]
        

        input_times = pd.date_range(start=input_start_date, end=input_end_date, periods=len(input_load))
        target_times = pd.date_range(start=pred_start_time+pd.Timedelta(minutes=15), periods=len(target_load), freq='15min')
        
        plt.plot(input_times, input_load, 'b-', label='Load Input Sequence', linewidth=1.5, alpha=0.8)
        plt.plot(target_times, target_load, 'g-o', label='Ground Truth Load', linewidth=2, markersize=4)
        plt.plot(target_times, pred_denormalized, 'r-o', label='Load Prediction', linewidth=2, markersize=4)
        
        line = plt.axvline(x=pred_start_time+pd.Timedelta(minutes=15),
                           color='gray', linewidth=2, alpha=0.7, label='Prediction Start')
        line.set_dashes((5, 4))  # (Strichlänge, Lückenlänge) in points — Werte anpassen
        
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Electrical Load [W]', fontsize=14)
        
        pred_start_time_pd = pd.Timestamp(pred_start_time+pd.Timedelta(minutes=15))
        plt.title(f'Load Prediction Sequence', 
                  fontsize=15, fontweight='bold')
        plt.legend(fontsize=14, loc='upper left')
        
        # Format x-axis as "April 15, 12:00"
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        #plt.xticks(rotation=30, ha='center')
        
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Save plot with datetime in filename (e.g. prediction_plot_2019-06-04_05-00.png)
        try:
            timestamp_str = pred_start_time_pd.strftime('%Y-%m-%d_%H-%M')
        except Exception:
            timestamp_str = time.strftime('%Y-%m-%d_%H-%M-%S')
        out_fname = f'prediction_plot_{timestamp_str}.png'
        plt.savefig(out_fname, dpi=300, bbox_inches='tight')
        
        if plot_metrics:
            # Calculate error
            mae = np.mean(np.abs(pred_denormalized - target_load))
            mse = np.mean((pred_denormalized - target_load) ** 2)
            
            textstr = f'MAE: {mae:.1f} W\nMSE: {mse:.1f} W²'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
            
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1) 

    print(f"{GREEN}Generated {n_examples} prediction plots!{RESET}")


def plot_full_test_set_predictions(model, test_loader, device="cpu", output_horizon=16, df_test=None, 
                                   seq_len=192, scaler=None, plot_metrics=False, 
                                   plot_prediction_window_indication=True,
                                   plot_integral_difference=False,
                                   integral_reset_timesteps=None,
                                   start_date=None,
                                   end_date=None):
    """
    Plottet Test-Predictions OHNE Überlappungen.
    Nimmt nur jeden output_horizon-ten Sample für eine echte kontinuierliche Timeline.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run inference on
        output_horizon: Output horizon (number of timesteps to predict) - bestimmt Sampling-Rate
        df_test: Original test dataframe (optional) - für echte Zeitachse
        seq_len: Sequence length (nur relevant wenn df_test gegeben)
        scaler: Scaler to denormalize data
        plot_metrics: Whether to display metrics in the plot
        plot_prediction_window_indication: Whether to show vertical lines between prediction windows
        plot_integral_difference: Whether to plot cumulative integral difference between prediction and ground truth
        integral_reset_timesteps: Number of timesteps after which the cumulative integral difference is reset. 
            At each reset, the cumulative sum restarts from the current value at that point, not from the start of the timeline.
            If None, defaults to output_horizon (reset after each forecast horizon). (default: None)
        start_date: Optional start date (str format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM') to filter plot range
        end_date: Optional end date (str format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM') to filter plot range
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []

    print(f"{CYAN}Generating Plot with predictions over the full test set...{RESET}")

    # Start timing
    start_time = time.time()

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_test.cpu().numpy())
            all_inputs.append(X_test.cpu().numpy())

    # End timing
    inference_time = time.time() - start_time


    all_preds = np.concatenate(all_preds, axis=0)      
    all_targets = np.concatenate(all_targets, axis=0) 
    all_inputs = np.concatenate(all_inputs, axis=0)    

    # Print timing information
    num_samples = len(all_preds)
    samples_per_second = num_samples / inference_time if inference_time > 0 else 0
    
    print(f"{GREEN}Prediction Performance:{RESET}")
    print(f"   Total prediction samples: {num_samples} (each predicts {output_horizon} timesteps ahead)")
    print(f"   Prediction time: {inference_time:.4f} seconds")
    print(f"   Throughput: {samples_per_second:.1f} samples/second")
    print(f"   Avg time per sample: {inference_time/num_samples*1000:.2f} ms")
    print()

    # Sample every output_horizon-th prediction to avoid overlaps
    # E.g., if seq_len=192 and output_horizon=16: Sample 0 predicts [192:208], Sample 16 predicts [208:224], etc.
    # E.g., if seq_len=192 and output_horizon=32: Sample 0 predicts [192:224], Sample 32 predicts [224:256], etc.
    # E.g., if seq_len=288 and output_horizon=16: Sample 0 predicts [288:304], Sample 16 predicts [304:320], etc.
    selected_indices = np.arange(0, len(all_preds), output_horizon)
    
    # print(f"   Total samples: {len(all_preds)}")
    # print(f"   Selected samples (every {output_horizon}th): {len(selected_indices)}")
    
    selected_preds = all_preds[selected_indices]      
    selected_targets = all_targets[selected_indices] 
    selected_inputs = all_inputs[selected_indices]    
    
    pred_timeline = selected_preds.flatten()
    target_timeline = selected_targets.flatten()
    
    # Extract load from inputs (LAST feature: [:, :, -1] = Load column)
    # For the first sample, we use all seq_len timesteps
    # For subsequent samples, we only use the last output_horizon timesteps to avoid overlap
    input_timeline_list = []
    
    input_timeline_list.append(selected_inputs[0, :, -1])  
    
    # Subsequent samples: only use last output_horizon timesteps to continue timeline
    for i in range(1, len(selected_inputs)):
        input_timeline_list.append(selected_inputs[i, -output_horizon:, -1])  
    
    input_timeline = np.concatenate(input_timeline_list)
    
    # Denormalize if scaler is provided
    if scaler is not None:
        # Denormalize if scaler is provided
        # Scaler fitted on different features depending on mode:
        # - Cyclic mode: ['Year', 'Load'] (2 features)
        # - Raw mode: ['Year', 'Month', 'Day', 'Weekday', 'Timestep', 'Load'] (6 features)
        # Load is always the LAST column in the scaler
        n_features = scaler.data_min_.shape[0]
        
        # Create dummy arrays: (n_samples, n_features) with only LAST column (Load) filled
        input_dummy = np.zeros((len(input_timeline), n_features))
        input_dummy[:, -1] = input_timeline  # Fill Load column
        input_timeline = scaler.inverse_transform(input_dummy)[:, -1]
        
        pred_dummy = np.zeros((len(pred_timeline), n_features))
        pred_dummy[:, -1] = pred_timeline  # Fill Load column
        pred_timeline = scaler.inverse_transform(pred_dummy)[:, -1]
        
        target_dummy = np.zeros((len(target_timeline), n_features))
        target_dummy[:, -1] = target_timeline  # Fill Load column
        target_timeline = scaler.inverse_transform(target_dummy)[:, -1]

        input_and_target_timeline = np.concatenate([input_timeline[:seq_len], target_timeline])

    
    print(f"Generated {len(selected_preds)} non-overlapping predictions")
    # print(f"   Input timesteps: {len(input_timeline)}")
    # print(f"   Prediction timesteps: {len(pred_timeline)}")
    # print(f"   Total timesteps: {len(input_timeline) + len(pred_timeline)}")

    
    
    if df_test is not None:

        input_time_indices = list(range(seq_len))
        
    
        for i in range(1, len(selected_indices)):
            sample_idx = selected_indices[i]
            start_idx = seq_len + sample_idx - output_horizon
            end_idx = seq_len + sample_idx
            if end_idx <= len(df_test):
                input_time_indices.extend(range(start_idx, end_idx))

        pred_time_indices = []
        for i, sample_idx in enumerate(selected_indices):
            start_time_idx = seq_len + sample_idx
            end_time_idx = start_time_idx + output_horizon
            if end_time_idx <= len(df_test):
                pred_time_indices.extend(range(start_time_idx, end_time_idx))
        
        # Get actual timestamps
        input_time_axis = df_test.index[input_time_indices[:len(input_timeline)]]
        pred_time_axis = df_test.index[pred_time_indices[:len(pred_timeline)]]
        input_and_target_axis = np.concatenate([input_time_axis[:seq_len], pred_time_axis])
        use_time_axis = True
        # print(f"   Input time range: {input_time_axis[0]} to {input_time_axis[-1]}")
        # print(f"   Prediction time range: {pred_time_axis[0]} to {pred_time_axis[-1]}")
        
        # Filter by date range if provided
        if start_date is not None or end_date is not None:
            # Parse dates
            if start_date is not None:
                start_dt = pd.Timestamp(start_date)
            else:
                start_dt = input_and_target_axis[0]
            
            if end_date is not None:
                end_dt = pd.Timestamp(end_date)
            else:
                end_dt = pred_time_axis[-1]
            
            # Filter input_and_target_timeline and pred_timeline by date range
            # Create masks for filtering
            input_target_mask = (input_and_target_axis >= start_dt) & (input_and_target_axis <= end_dt)
            pred_mask = (pred_time_axis >= start_dt) & (pred_time_axis <= end_dt)
            
            # Apply masks
            input_and_target_axis = input_and_target_axis[input_target_mask]
            input_and_target_timeline = input_and_target_timeline[input_target_mask]
            pred_time_axis = pred_time_axis[pred_mask]
            pred_timeline = pred_timeline[pred_mask]
            target_timeline = target_timeline[pred_mask]
            
            print(f"{CYAN}Filtered to date range: {start_dt} to {end_dt}{RESET}")
            print(f"   Filtered timesteps: {len(pred_timeline)} prediction timesteps")
        
    else:
        input_and_target_axis = np.arange(len(input_and_target_timeline))
        pred_time_axis = np.arange(len(input_and_target_axis) - len(pred_timeline), len(input_and_target_timeline))
        
        use_time_axis = False


    # Create figure - with subplots if integral difference is requested
    if plot_integral_difference:
        fig, (ax, ax_int) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(20, 6))


    if use_time_axis:
        ax.plot(input_and_target_axis, input_and_target_timeline, 'b-', label='Ground Truth Load', alpha=0.8, linewidth=1.8)
        ax.plot(pred_time_axis, pred_timeline, 'r-', label='Load Prediction(new Pred. all 12h)', alpha=0.8, linewidth=1.8)  
        ax.set_xlabel("Time", fontsize=13)
        
        # Format x-axis with date formatter
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
        ax.tick_params(axis='x', labelsize=11)
        #plt.xticks(rotation=45, ha='right')
    else:
        ax.plot(input_and_target_axis, input_and_target_timeline, 'g-', label='Ground Truth Load', alpha=0.8, linewidth=1.8)
        ax.plot(pred_time_axis, pred_timeline, 'r-', label='Load Prediction (New Pred. all 12h)', alpha=0.8, linewidth=1.8)
        ax.set_xlabel("Timestep (kontinuierliche Timeline - OHNE Überlappungen)", fontsize=13)
        
        
    
    # Y-axis label depends on whether data is normalized
    ylabel = "Electrical Load [W]" if scaler is not None else "Normalized Load"
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Load Prediction Sequences over 10 Days", 
              fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Vertical prediction window lines disabled
    # if plot_prediction_window_indication:
    #     # Add vertical lines to separate prediction windows
    #     # Each window starts at seq_len + i * output_horizon
    #     for i in range(len(selected_preds)):
    #         if use_time_axis:
    #             # Use actual timestamps
    #             window_start_idx = seq_len + i * output_horizon
    #             if window_start_idx < len(input_and_target_axis):
    #                 ax.axvline(x=input_and_target_axis[window_start_idx], 
    #                         color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    #         else:
    #             # Use timestep indices
    #             window_start_idx = seq_len + i * output_horizon
    #             ax.axvline(x=window_start_idx, 
    #                     color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    
    if plot_metrics:
        mse = np.mean((pred_timeline - target_timeline) ** 2)
        mae = np.mean(np.abs(pred_timeline - target_timeline))
        rmse = np.sqrt(mse)
        
        textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\n(ohne Überlappungen)'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
    # Plot cumulative integral difference if requested (as second subplot)
    if plot_integral_difference:
        print(f"{CYAN}Generating cumulative integral difference plot...{RESET}")
        
        # Set default reset length if not specified
        if integral_reset_timesteps is None:
            integral_reset_timesteps = output_horizon
        
        # Compute cumulative integrals with resets
        # Each timestep is 15 minutes = 0.25 hours, so multiply by 0.25 to get Wh
        timestep_hours = 0.25  # 15 minutes in hours
        
        cumsum_pred = np.zeros_like(pred_timeline)
        cumsum_target = np.zeros_like(target_timeline)
        
        reset_length = integral_reset_timesteps
        
        for i in range(len(pred_timeline)):
            # Reset to 0 at the beginning of each reset interval
            if i % reset_length == 0:
                cumsum_pred[i] = pred_timeline[i] * timestep_hours
                cumsum_target[i] = target_timeline[i] * timestep_hours
            else:
                cumsum_pred[i] = cumsum_pred[i-1] + pred_timeline[i] * timestep_hours
                cumsum_target[i] = cumsum_target[i-1] + target_timeline[i] * timestep_hours
        
        # Compute difference of integrals
        integral_diff = cumsum_pred - cumsum_target
        
        # Plot in second subplot (ax_int already created above)
        
        if use_time_axis:
            ax_int.plot(pred_time_axis, integral_diff, 'purple', label='Cumulative Integral Difference', 
                       alpha=0.8, linewidth=1.5)
            ax_int.set_xlabel("Time", fontsize=13)
            # Format x-axis with date formatter
            ax_int.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_int.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
            ax_int.tick_params(axis='x', labelsize=11)
        else:
            ax_int.plot(pred_time_axis, integral_diff, 'purple', label='Cumulative Integral Difference', 
                       alpha=0.8, linewidth=1.5)
            ax_int.set_xlabel("Timestep (kontinuierliche Timeline - OHNE Überlappungen)", fontsize=12)
        
        # Add zero reference line
        ax_int.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Add vertical lines at reset points
        for i in range(0, len(pred_timeline), reset_length):
            if use_time_axis:
                if i < len(pred_time_axis):
                    ax_int.axvline(x=pred_time_axis[i], color='red', linestyle='--', 
                                  linewidth=1.0, alpha=0.3)
            else:
                ax_int.axvline(x=pred_time_axis[i], color='red', linestyle='--', 
                              linewidth=1.0, alpha=0.3)
        
        ylabel_int = "Cumulative Integral Difference [Wh]" if scaler is not None else "Cumulative Integral Difference"
        ax_int.set_ylabel(ylabel_int, fontsize=12)
        
        if integral_reset_timesteps == output_horizon:
            reset_info = "(Reset after each prediction horizon)"
        else:
            reset_info = f"(Reset every {integral_reset_timesteps} timesteps = {integral_reset_timesteps/4:.1f} hours)"
        ax_int.set_title(f"Cumulative Integral Difference: ∫(Prediction - Ground Truth) {reset_info}", 
                        fontsize=14, fontweight='bold')
        ax_int.legend(fontsize=13, loc='upper right')
        ax_int.grid(True, alpha=0.3)
        
        # Calculate average integral difference at the end of each prediction horizon
        # Collect values at every output_horizon-th timestep
        horizon_end_indices = np.arange(output_horizon - 1, len(integral_diff), output_horizon)
        horizon_end_values = integral_diff[horizon_end_indices]
        avg_horizon_end_diff = np.mean(np.abs(horizon_end_values))
        
        # Add statistics
        mean_abs_diff = np.mean(np.abs(integral_diff))
        max_abs_diff = np.max(np.abs(integral_diff))
        
        textstr_int = (f'Mean |Integral Diff|: {mean_abs_diff:.2f} Wh\n'
                      f'Max |Integral Diff|: {max_abs_diff:.2f} Wh\n'
                      f'Avg |Diff| at Horizon End: {avg_horizon_end_diff:.2f} Wh')
        props_int = dict(boxstyle='round', facecolor='lavender', alpha=0.8)
        ax_int.text(0.02, 0.98, textstr_int, transform=ax_int.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props_int)
        
        print(f"{GREEN}Cumulative integral difference plot added!{RESET}")
    
    # Show the combined figure (either single plot or both plots)
    plt.tight_layout()
    
    # X-axis labels without rotation
    # if use_time_axis:
    #     plt.setp(fig.axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save plot with datetime in filename
    if use_time_axis and len(pred_time_axis) > 0:
        try:
            first_pred_time = pd.Timestamp(pred_time_axis[0])
            timestamp_str = first_pred_time.strftime('%Y-%m-%d_%H-%M')
        except Exception:
            timestamp_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        timestamp_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    out_fname = f'full_test_set_plot_{timestamp_str}.png'
    plt.savefig(out_fname, dpi=300, bbox_inches='tight')
    
    plt.show(block=False)
    plt.pause(0.1)
    
    print(f"{GREEN}Full test set plot generated and saved to {out_fname}!{RESET}")

    return selected_preds, selected_targets