import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_predictions(model, test_loader, device="cpu", n_examples=1):
    """
    Plots example predictions from the test set.
    Shows input sequence, true output, and predicted output.
    """
    print(f"{CYAN}Generating one prediction from test set...{RESET}")
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)

            for j in range(min(n_examples, X_test.size(0))):
                plt.figure(figsize=(12, 4))
                input_seq = X_test[j][:, -1].cpu().numpy()
                target_seq = y_test[j].cpu().numpy()
                pred_seq = y_pred[j].cpu().numpy()

                plt.plot(range(len(input_seq)), input_seq, label='Input sequence', color='blue')
                plt.plot(range(len(input_seq), len(input_seq) + len(target_seq)),
                         target_seq, 'r-o', label='True target')
                plt.plot(range(len(input_seq), len(input_seq) + len(pred_seq)),
                         pred_seq, 'g--x', label='Prediction')

                plt.xlabel("Timestep (15-min intervals)")
                plt.ylabel("Normalized Load")
                plt.title(f"Example Prediction {j+1}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1) 

            break

    print(f"{GREEN}Generated one prediction plot!{RESET}")


def plot_raw_dataframe(df, title="Test Data"):
    """
    Simply plots the raw DataFrame time series data (only 'Load' column).
    Can also plot multiple dataframes for train/val/test comparison.
    
    Args:
        df: pandas DataFrame with time series data, or tuple of (df_train, df_val, df_test)
        title: Plot title
    """
    # Check if we have multiple dataframes
    if isinstance(df, tuple) and len(df) == 3:
        df_train, df_val, df_test = df

        print(f"{CYAN} Plotting train/val/test split...{RESET}")
        print(f"   Train: {len(df_train)} samples ({df_train.index[0]} to {df_train.index[-1]})")
        print(f"   Val:   {len(df_val)} samples ({df_val.index[0]} to {df_val.index[-1]})")
        print(f"   Test:  {len(df_test)} samples ({df_test.index[0]} to {df_test.index[-1]})")
        
        # Check for Load column
        if 'Load' not in df_train.columns:
            print(f"Warning: 'Load' column not found. Using first column")
            load_col = df_train.columns[0]
        else:
            load_col = 'Load'
        
        # Create figure
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
            
            # Statistics
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


def plot_multiple_predictions_at_date(model, df_test, start_date = None, n_examples=5, seq_len=192, output_horizon=16, scaler=None, device="cpu"):
    """
    Plot multiple predictions starting at a given date, spaced 12 hours (half day) apart.
    
    Args:
        model: Trained model
        df_test: Test dataframe
        start_date: Starting date string (e.g., '2018-08-31') or datetime object
        n_examples: Number of predictions to plot (each 12 hours apart)
        seq_len: Sequence length for input
        output_horizon: Prediction horizon
        scaler: MinMaxScaler for normalization
        device: Device for model inference
    """

    print(f"{CYAN}Generating {n_examples} predictions from test set...{RESET}")
    
    # 12 hours = 48 timesteps (at 15-minute intervals)
    spacing_timesteps = 48
    
    # Handle start_date
    if start_date is None:
        # Use earliest possible date (need seq_len history)
        closest_idx = seq_len
        actual_start = df_test.index[closest_idx]
        print(f"   No start date provided. Using earliest possible date: {actual_start}")
        print(f"   Generating {n_examples} predictions spaced 12 hours apart...")
    else:
        print(f"   Generating {n_examples} predictions starting from {start_date}, spaced 12 hours apart...")

        # Convert to datetime if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Find starting index
        if start_date not in df_test.index:
            closest_idx = df_test.index.get_indexer([start_date], method='nearest')[0]
            actual_start = df_test.index[closest_idx]
            print(f"Exact date not found. Using closest date: {actual_start}")
        else:
            actual_start = start_date
            closest_idx = df_test.index.get_loc(actual_start)
    
    # Check if we have enough data
    last_needed_idx = closest_idx + (n_examples - 1) * spacing_timesteps + output_horizon
    if closest_idx < seq_len:
        print(f" Error: Not enough history. Need {seq_len} timesteps before {actual_start}")
        return
    if last_needed_idx > len(df_test):
        print(f" Error: Not enough data for {n_examples} predictions")
        print(f"  Try reducing n_examples or choosing an earlier start date")
        return
    
    
    
    for i in range(n_examples):
        pred_start_idx = closest_idx + i * spacing_timesteps
        
        # Extract sequences
        start_idx = pred_start_idx - seq_len
        end_idx = pred_start_idx
        future_idx = pred_start_idx + output_horizon
        
        input_data = df_test.iloc[start_idx:end_idx]
        target_data = df_test.iloc[end_idx:future_idx]
        pred_start_time = df_test.index[pred_start_idx]
        
        print(f"   [{i+1}/{n_examples}] Prediction at {pred_start_time}")
        
        # Normalize
        if scaler is not None:
            input_normalized = scaler.transform(input_data.values)
            target_normalized = scaler.transform(target_data.values)
        else:
            input_normalized = input_data.values
            target_normalized = target_data.values
        

        X = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            y_pred = model(X)
            pred_normalized = y_pred.cpu().numpy()[0]
        
        # Denormalize
        if scaler is not None:
            pred_full = np.zeros((len(pred_normalized), input_data.shape[1]))
            pred_full[:, -1] = pred_normalized
            pred_denormalized = scaler.inverse_transform(pred_full)[:, -1]
        else:
            pred_denormalized = pred_normalized
        
        # Plot
        plt.figure(figsize=(16, 6))
        
        # Get Load column
        input_load = input_data['Load'].values if 'Load' in input_data.columns else input_data.iloc[:, -1].values
        target_load = target_data['Load'].values if 'Load' in target_data.columns else target_data.iloc[:, -1].values
        
        # Create time axis
        input_times = input_data.index
        target_times = target_data.index
        
        # Plot
        plt.plot(input_times, input_load, 'b-', label='Input Sequence (Historical)', linewidth=1.5, alpha=0.8)
        plt.plot(target_times, target_load, 'g-o', label='Ground Truth', linewidth=2, markersize=4)
        plt.plot(target_times, pred_denormalized, 'r--x', label='Prediction', linewidth=2, markersize=5)
        plt.axvline(x=pred_start_time, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Prediction Start')
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Load [kW]', fontsize=12)
        plt.title(f'Prediction {i+1}/{n_examples} - Starting at {pred_start_time.strftime("%Y-%m-%d %H:%M")}', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Calculate error
        mae = np.mean(np.abs(pred_denormalized - target_load))
        mse = np.mean((pred_denormalized - target_load) ** 2)
        
        textstr = f'MAE: {mae:.3f} kW\nMSE: {mse:.3f} kW²'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure plot displays

    print(f"{GREEN}Generated {n_examples} prediction plots!{RESET}")


def plot_full_test_set_predictions(model, test_loader, device="cpu", output_horizon=16, df_test=None, seq_len=192, scaler=None):
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

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)      # Shape: (n_samples, output_horizon)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples, output_horizon)
    all_inputs = np.concatenate(all_inputs, axis=0)    # Shape: (n_samples, seq_len, input_dim)

    # Sample every output_horizon-th prediction to avoid overlaps
    # Sample 0 predicts [192:208], Sample 16 predicts [208:224], Sample 32 predicts [224:240], etc.
    selected_indices = np.arange(0, len(all_preds), output_horizon)
    
    print(f"   Total samples: {len(all_preds)}")
    print(f"   Selected samples (every {output_horizon}th): {len(selected_indices)}")
    
    # Select only non-overlapping samples
    selected_preds = all_preds[selected_indices]      # Shape: (n_selected, output_horizon)
    selected_targets = all_targets[selected_indices]  # Shape: (n_selected, output_horizon)
    selected_inputs = all_inputs[selected_indices]    # Shape: (n_selected, seq_len, input_dim)
    
    # Now flatten to create TRUE continuous timeline (no overlaps!)
    pred_timeline = selected_preds.flatten()
    target_timeline = selected_targets.flatten()
    
    # Extract load from inputs (LAST feature: [:, :, -1] = Load column)
    # For the first sample, we use all seq_len timesteps
    # For subsequent samples, we only use the last output_horizon timesteps to avoid overlap
    input_timeline_list = []
    
    # First sample: use full input sequence
    input_timeline_list.append(selected_inputs[0, :, -1])  # All seq_len timesteps, Load is last column
    
    # Subsequent samples: only use last output_horizon timesteps to continue timeline
    for i in range(1, len(selected_inputs)):
        input_timeline_list.append(selected_inputs[i, -output_horizon:, -1])  # Load is last column
    
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
        input_dummy[:, -1] = input_timeline  # Load is last column (index 5 or -1)
        input_timeline = scaler.inverse_transform(input_dummy)[:, -1]
        
        pred_dummy = np.zeros((len(pred_timeline), n_features))
        pred_dummy[:, -1] = pred_timeline  # Load is last column
        pred_timeline = scaler.inverse_transform(pred_dummy)[:, -1]
        
        target_dummy = np.zeros((len(target_timeline), n_features))
        target_dummy[:, -1] = target_timeline  # Load is last column
        target_timeline = scaler.inverse_transform(target_dummy)[:, -1]

        input_and_target_timeline = np.concatenate([input_timeline[:seq_len], target_timeline])

        print(f"   Data denormalized to original scale")
        print(f"   Input range: [{input_timeline.min():.2f}, {input_timeline.max():.2f}]")
        print(f"   Prediction range: [{pred_timeline.min():.2f}, {pred_timeline.max():.2f}]")
    
    print(f"Generated {len(selected_preds)} non-overlapping predictions")
    print(f"   Input timesteps: {len(input_timeline)}")
    print(f"   Prediction timesteps: {len(pred_timeline)}")
    print(f"   Total timesteps: {len(input_timeline) + len(pred_timeline)}")
    
    # Create time axis if df_test is provided
    
    
    if df_test is not None:
        # Input time indices: start from 0 in df_test
        input_time_indices = list(range(seq_len))
        
        # Add continuation of input for each subsequent sample
        for i in range(1, len(selected_indices)):
            sample_idx = selected_indices[i]
            start_idx = seq_len + sample_idx - output_horizon
            end_idx = seq_len + sample_idx
            if end_idx <= len(df_test):
                input_time_indices.extend(range(start_idx, end_idx))
        
        # Prediction time indices
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
    
    # Calculate and display metrics
    mse = np.mean((pred_timeline - target_timeline) ** 2)
    mae = np.mean(np.abs(pred_timeline - target_timeline))
    rmse = np.sqrt(mse)
    
    # Add text box with metrics
    textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}\n(ohne Überlappungen)'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to ensure plot displays

    print(f"{GREEN}Full test set plot generated!{RESET}")

    return selected_preds, selected_targets