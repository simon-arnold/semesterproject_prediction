import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_processing.EnergyDataset import EnergyDataset
from sklearn.preprocessing import MinMaxScaler

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

        fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=False)
        
        datasets = [
            (df_train, 'Train', 'blue'),
            (df_val, 'Validation', 'orange'),
            (df_test, 'Test', 'green')
        ]
        
        for i, (data, label, color) in enumerate(datasets):
            axes[i].plot(data.index, data[load_col].values, linewidth=0.8, color=color, alpha=0.8)
            axes[i].set_ylabel(f'{load_col} [kW]', fontsize=11)
            axes[i].set_title(f"{label} Dataset ({len(data)} samples)", fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            

            mean_val = data[load_col].mean()
            std_val = data[load_col].std()
            min_val = data[load_col].min()
            max_val = data[load_col].max()
            
            textstr = f'Mean: {mean_val:.3f} kW | Std: {std_val:.3f} kW | Min: {min_val:.3f} kW | Max: {max_val:.3f} kW'
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
        plt.ylabel(f'{load_col} [kW]', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.title(f"{title} - {load_col} Time Series ({len(df)} samples)", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = df[load_col].mean()
        std_val = df[load_col].std()
        min_val = df[load_col].min()
        max_val = df[load_col].max()
        
        textstr = f'Mean: {mean_val:.3f} kW\nStd: {std_val:.3f} kW\nMin: {min_val:.3f} kW\nMax: {max_val:.3f} kW'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()

    print(f"{GREEN}Train/data/test plot generated!{RESET}")


def plot_test_data_overview(test_loader):
    """
    Plots only the ground truth test data in a continuous timeline.
    No predictions - just shows what data we're testing on.
    
    Args:
        test_loader: DataLoader for test set
    """
    all_targets = []
    
    print("Loading test data for visualization...")
    
    for X_test, y_test in test_loader:
        all_targets.append(y_test.cpu().numpy())
    
    # Concatenate all targets
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples, output_horizon)
    
    # Flatten to create continuous timeline
    target_timeline = all_targets.flatten()
    
    print(f"Loaded {len(all_targets)} test samples ({len(target_timeline)} timesteps total)")
    
    # Create figure
    plt.figure(figsize=(20, 6))
    
    timesteps = np.arange(len(target_timeline))
    
    plt.plot(timesteps, target_timeline, 'b-', label='Test Data (Ground Truth)', alpha=0.8, linewidth=1)
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Normalized Load", fontsize=12)
    plt.title(f"Complete Test Dataset Overview ({len(all_targets)} samples, {len(target_timeline)} timesteps)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add statistics text box
    mean_val = np.mean(target_timeline)
    std_val = np.std(target_timeline)
    min_val = np.min(target_timeline)
    max_val = np.max(target_timeline)
    
    textstr = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.show()
    
    print(f"Test data overview plot generated!")
    
    return all_targets


def plot_multiple_predictions_at_date(model, test_set: EnergyDataset, start_date = None, 
                                      n_examples=5, seq_len=192, output_horizon=16, device="cpu", 
                                      plot_metrics=False):
    """
    Plot multiple predictions starting at a given date, spaced 12 hours (half day) apart.
    
    Args:
        model: Trained model
        df_test: Test dataframe
        start_date: Last Date of the Input Sequence.
        n_examples: Number of predictions to plot (each 12 hours apart)
        seq_len: Sequence length for input
        output_horizon: Prediction horizon
        scaler: MinMaxScaler for normalization
        device: Device for model inference
    """
    #TODO: Das naming mit start_date und pred_start_date stimmt nicht ganz mit dem was es wirklich ist überein. 
    # Also die funktion macht das richtige. Aber pred_start_time und dessen index sollten ja sicher um 1 oder 
    # 15 minuten verschoben sein zu dem letzen datum des inputs(start_time). Diese Verschiebung habe ich mit 
    # dem 15 minuten offsett im print geregelt. In der logik aber nicht - die müsste noch angepasst werden 
    # der Vollständigkeit zu liebe.

    print(f"{CYAN}Generating {n_examples} predictions from test set...{RESET}")
    
    # Generating a plot after each complete forecast horizon.
    spacing_timesteps = output_horizon
    
    # Handle start_date
    if start_date is None:
        
        # Use earliest possible date (need seq_len history)
        actual_start = test_set.input_end_dates[0]
        print(f"   No start date provided. Using earliest possible date: {actual_start+pd.Timedelta(minutes=15)}")
        print(f"   Generating {n_examples} predictions spaced 8 hours apart...")
    else:
        print(f"   Generating {n_examples} predictions starting from {start_date}, spaced 8 hours apart...")

        if isinstance(start_date, str):
            start_date = np.datetime64(start_date) - np.timedelta64(15, 'm')
        else:
            raise NotImplementedError(
                "Es ist nicht klar was passiert wenn start_date input kein string ist "
                "- gilt es zu implementieren und überprüfen, vorallem mit dem 1 oder 15 min index/time offset."
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
        n_features = test_set.scaler.data_min_.shape[0] 
        
        pred_full = np.zeros((len(pred_normalized[0]), n_features))
        pred_full[:, -1] = pred_normalized[0]  
        pred_denormalized = test_set.scaler.inverse_transform(pred_full)[:, -1]
        
        input_full = np.zeros((len(test_set.X[pred_start_idx]), n_features))
        input_full[:, -1] = test_set.X[pred_start_idx][:, -1].cpu().numpy()
        input_load_denormalized = test_set.scaler.inverse_transform(input_full)[:, -1]
        
        target_full = np.zeros((len(test_set.y[pred_start_idx]), n_features))
        target_full[:, -1] = test_set.y[pred_start_idx].cpu().numpy()  
        target_load_denormalized = test_set.scaler.inverse_transform(target_full)[:, -1]
               
        # Plot
        plt.figure(figsize=(16, 6))
        
        input_load = input_load_denormalized
        target_load = target_load_denormalized
        
        input_start_date = test_set.input_start_dates[pred_start_idx]
        input_end_date = test_set.input_end_dates[pred_start_idx]
        

        input_times = pd.date_range(start=input_start_date, end=input_end_date, periods=len(input_load))
        target_times = pd.date_range(start=pred_start_time+pd.Timedelta(minutes=15), periods=len(target_load), freq='15min')
        
        plt.plot(input_times, input_load, 'b-', label='Input Sequence (Historical)', linewidth=1.5, alpha=0.8)
        plt.plot(target_times, target_load, 'g-o', label='Ground Truth', linewidth=2, markersize=4)
        plt.plot(target_times, pred_denormalized, 'r--x', label='Prediction', linewidth=2, markersize=5)
        plt.axvline(x=pred_start_time+pd.Timedelta(minutes=15), color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Prediction Start')
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Load [kW]', fontsize=12)
        
        pred_start_time_pd = pd.Timestamp(pred_start_time+pd.Timedelta(minutes=15))
        plt.title(f'Prediction {i+1}/{n_examples} - Starting at {pred_start_time_pd.strftime("%Y-%m-%d %H:%M")}', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        if plot_metrics:
            # Calculate error
            mae = np.mean(np.abs(pred_denormalized - target_load))
            mse = np.mean((pred_denormalized - target_load) ** 2)
            
            textstr = f'MAE: {mae:.3f} kW\nMSE: {mse:.3f} kW²'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=props)
            
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1) 

    print(f"{GREEN}Generated {n_examples} prediction plots!{RESET}")


def plot_full_test_set_predictions(model, test_loader, device="cpu", output_horizon=16, df_test=None, 
                                   seq_len=192, scaler=None, plot_metrics=False, 
                                   plot_prediction_window_indication=True):
    """
    Plottet Test-Predictions OHNE Überlappungen.
    Nimmt nur jeden output_horizon-ten Sample für eine echte kontinuierliche Timeline.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run inference on
        output_horizon: Output horizon (z.B. 16) - bestimmt Sampling-Rate
        df_test: Original test dataframe (optional) - für echte Zeitachse
        seq_len: Sequence length (nur relevant wenn df_test gegeben)
        scaler: Scaler to denormalize data
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []

    print(f"{CYAN}Generating Plot with predictions over the full test set...{RESET}")

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_test.cpu().numpy())
            all_inputs.append(X_test.cpu().numpy())


    all_preds = np.concatenate(all_preds, axis=0)      
    all_targets = np.concatenate(all_targets, axis=0) 
    all_inputs = np.concatenate(all_inputs, axis=0)    

    # Sample every output_horizon-th prediction to avoid overlaps
    # Sample 0 predicts [192:208], Sample 16 predicts [208:224], Sample 32 predicts [224:240], etc.
    selected_indices = np.arange(0, len(all_preds), output_horizon)
    
    print(f"   Total samples: {len(all_preds)}")
    print(f"   Selected samples (every {output_horizon}th): {len(selected_indices)}")
    
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
        print(f"Denormalizing data...")
        # Denormalize input, predictions, and targets
        # Scaler expects shape (n_samples, n_features=6), but we only want to denormalize load (feature 0)
        # Create dummy arrays with zeros for other features
        n_features = scaler.data_min_.shape[0]  # Should be 6
        
        # Create dummy arrays: (n_samples, n_features) with only LAST column (Load) filled
        input_dummy = np.zeros((len(input_timeline), n_features))
        input_dummy[:, -1] = input_timeline 
        input_timeline = scaler.inverse_transform(input_dummy)[:, -1]
        
        pred_dummy = np.zeros((len(pred_timeline), n_features))
        pred_dummy[:, -1] = pred_timeline  
        pred_timeline = scaler.inverse_transform(pred_dummy)[:, -1]
        
        target_dummy = np.zeros((len(target_timeline), n_features))
        target_dummy[:, -1] = target_timeline  
        target_timeline = scaler.inverse_transform(target_dummy)[:, -1]

        input_and_target_timeline = np.concatenate([input_timeline[:seq_len], target_timeline])

        print(f"   Data denormalized to original scale")
        print(f"   Input range: [{input_timeline.min():.2f}, {input_timeline.max():.2f}]")
        print(f"   Prediction range: [{pred_timeline.min():.2f}, {pred_timeline.max():.2f}]")
    
    print(f"Generated {len(selected_preds)} non-overlapping predictions")
    print(f"   Input timesteps: {len(input_timeline)}")
    print(f"   Prediction timesteps: {len(pred_timeline)}")
    print(f"   Total timesteps: {len(input_timeline) + len(pred_timeline)}")

    
    
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
        print(f"   Input time range: {input_time_axis[0]} to {input_time_axis[-1]}")
        print(f"   Prediction time range: {pred_time_axis[0]} to {pred_time_axis[-1]}")
        
    else:
        input_and_target_axis = np.arange(len(input_and_target_timeline))
        pred_time_axis = np.arange(len(input_and_target_axis) - len(pred_timeline), len(input_and_target_timeline))
        
        use_time_axis = False


    # Create figure
    fig, ax = plt.subplots(figsize=(20, 6))
    
    print("input_and_target_timeline shape:", np.shape(input_and_target_timeline))
    print("pred_time_axis shape:", np.shape(pred_time_axis))

    if use_time_axis:
        ax.plot(input_and_target_axis, input_and_target_timeline, 'b-', label='Ground Truth', alpha=0.8, linewidth=1.2)
        ax.plot(pred_time_axis, pred_timeline, 'r-', label='Predictions', alpha=0.8, linewidth=1.2)  
        ax.set_xlabel("Time", fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
    else:
        ax.plot(input_and_target_axis, input_and_target_timeline, 'b-', label='Ground Truth', alpha=0.8, linewidth=1.2)
        ax.plot(pred_time_axis, pred_timeline, 'r-', label='Predictions', alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Timestep (kontinuierliche Timeline - OHNE Überlappungen)", fontsize=12)
        
        
    
    # Y-axis label depends on whether data is normalized
    ylabel = "Load [kW]" if scaler is not None else "Normalized Load"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Full Test Set: Input + Non-Overlapping Predictions ({len(selected_preds)} samples, {len(input_timeline)} input + {len(pred_timeline)} prediction steps)", 
              fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    
    if plot_prediction_window_indication:
        # Add vertical lines to separate prediction windows
        # Each window starts at seq_len + i * output_horizon
        for i in range(len(selected_preds)):
            if use_time_axis:
                # Use actual timestamps
                window_start_idx = seq_len + i * output_horizon
                if window_start_idx < len(input_and_target_axis):
                    ax.axvline(x=input_and_target_axis[window_start_idx], 
                            color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
            else:
                # Use timestep indices
                window_start_idx = seq_len + i * output_horizon
                ax.axvline(x=window_start_idx, 
                        color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    
    if plot_metrics:
        mse = np.mean((pred_timeline - target_timeline) ** 2)
        mae = np.mean(np.abs(pred_timeline - target_timeline))
        rmse = np.sqrt(mse)
        
        textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\n(ohne Überlappungen)'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1) 

    print(f"{GREEN}Full test set plot generated!{RESET}")

    return selected_preds, selected_targets