"""
Compare multiple saved models with different forecast horizons.

This script loads multiple models and plots their predictions in a single plot
for comparison.

Example usage:
    python src/compare_models.py --plot_full --plot_integrated_difference
"""

import sys
import os
import importlib.util
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from torch.utils.data import DataLoader
from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe
from data_processing.data_utils_2nd_house import load_energy_hdf_to_pandas_2nd_house

# ANSI Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BRIGHT_BLUE = '\033[94;1m'
CYAN = '\033[96m'
RESET = '\033[0m'

PRINT_DEBUG = False

HOUSE_TYPE = 'A'  # Options: 'A' or 'E' (2nd house)

# Model configurations
MODELS = {
    'hor_16_A': {
        'name': '16-step (no temp) A',
        'path': 'NN_storage_good_model/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'NN_storage_good_model/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': False,
        'color': 'black'
    },
    'hor_16_temp_A': {
        'name': '16-step A',
        'path': 'house_A/NN_storage_hor_16/NN_storage_3/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'house_A/NN_storage_hor_16/NN_storage_3/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_16_A_ker5': {
        'name': '16-step A',
        'path': 'house_A/NN_storage_hor_16/NN_storage_4_ker5/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'house_A/NN_storage_hor_16/NN_storage_4_ker5/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_24_A': {
        'name': '24-step A',
        'path': 'house_A/NN_storage_hor_24/NN_storage_5/NN_weights/best_model.pth',
        'model_file': 'house_A/NN_storage_hor_24/NN_storage_5/cnn_lstm.py',
        'horizon': 24,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'green'
    },
    'hor_32_A': {
        'name': '32-step A',
        'path': 'house_A/NN_storage_hor_32/NN_storage_8/NN_weights/best_model.pth',
        'model_file': 'house_A/NN_storage_hor_32/NN_storage_8/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
    'hor_32_A_24_3': {
        'name': '32-step A 24 3',
        'path': 'house_A/NN_storage_hor_32/NN_storage_11_24er/NN_weights/best_model.pth',
        'model_file': 'house_A/NN_storage_hor_32/NN_storage_11_24er/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_48_A': {
        'name': '48-step A',
        'path': 'house_A/NN_storage_hor_48/NN_storage_3/NN_weights/best_model.pth',
        'model_file': 'house_A/NN_storage_hor_48/NN_storage_3/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },
    'hor_48_A_24_5': {
        'name': 'Load Prediction(new Pred. all 12h)',
        'path': 'house_A/NN_storage_hor_48/NN_storage_8_24er/NN_weights/best_model.pth',
        'model_file': 'house_A/NN_storage_hor_48/NN_storage_8_24er/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },
    'hor_16_A_no_HP': {
        'name': '16-step A (no HP)',
        'path': 'house_A_wo_HP/NN_storage_hor_16/NN_storage_2/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_16/NN_storage_2/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
   'hor_16_A_no_HP_ker5': {
        'name': '16-step A (no HP)',
        'path': 'house_A_wo_HP/NN_storage_hor_16/NN_storage_3_ker5/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_16/NN_storage_3_ker5/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_24_A_no_HP': {
        'name': '24-step A (no HP)',
        'path': 'house_A_wo_HP/NN_storage_hor_24/NN_storage_2/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_24/NN_storage_2/cnn_lstm.py',
        'horizon': 24,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'green'
    },
    'hor_32_A_no_HP': {
        'name': '32-step A (no HP)',
        'path': 'house_A_wo_HP/NN_storage_hor_32/NN_storage_1/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_32/NN_storage_1/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
    'hor_32_A_no_HP_24_1': {
        'name': '32-step A (no HP) 24 1',
        'path': 'house_A_wo_HP/NN_storage_hor_32/NN_storage_3_24er/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_32/NN_storage_3_24er/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
    'hor_48_A_no_HP': {
        'name': '48-step A (no HP)',
        'path': 'house_A_wo_HP/NN_storage_hor_48/NN_storage_1/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_48/NN_storage_1/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },
    'hor_48_A_no_HP_24_2': {
        'name': 'Load Prediction(new Pred. all 12h)',
        'path': 'house_A_wo_HP/NN_storage_hor_48/NN_storage_4_24er/NN_weights/best_model.pth',
        'model_file': 'house_A_wo_HP/NN_storage_hor_48/NN_storage_4_24er/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },
   
    'hor_16_E': {
        'name': '16-step E',
        'path': 'house_E/NN_storage_hor_16/NN_storage_1/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'house_E/NN_storage_hor_16/NN_storage_1/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_16_E_ker5': {
        'name': 'Load Prediction(new Pred. all 4h)',
        'path': 'house_E/NN_storage_hor_16/NN_storage_4_ker5/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'house_E/NN_storage_hor_16/NN_storage_4_ker5/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_24_E': {
        'name': 'Load Prediction(new Pred. all 6h)',
        'path': 'house_E/NN_storage_hor_24/NN_storage_1/NN_weights/best_model.pth',
        'model_file': 'house_E/NN_storage_hor_24/NN_storage_1/cnn_lstm.py',
        'horizon': 24,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'green'
    },
        'hor_32_E_24_2': {
        'name': 'Load Prediction(new Pred. all 8h)',
        'path': 'house_E/NN_storage_hor_32/NN_storage_6_24er/NN_weights/best_model.pth',
        'model_file': 'house_E/NN_storage_hor_32/NN_storage_6_24er/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
        'hor_32_E': {
        'name': 'Load Prediction(new Pred. all 8h)',
        'path': 'house_E/NN_storage_hor_32/NN_storage_3/NN_weights/best_model.pth',
        'model_file': 'house_E/NN_storage_hor_32/NN_storage_3/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
        'hor_48_E': {
        'name': 'Load Prediction(new Pred. all 12h)',
        'path': 'house_E/NN_storage_hor_48/NN_storage_3/NN_weights/best_model.pth',
        'model_file': 'house_E/NN_storage_hor_48/NN_storage_3/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },
        'hor_48_E_24_2': {
        'name': 'Load Prediction(new Pred. all 12h)',
        'path': 'house_E/NN_storage_hor_48/NN_storage_5_24er/NN_weights/best_model.pth',
        'model_file': 'house_E/NN_storage_hor_48/NN_storage_5_24er/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    },


}


def load_model_class(model_file_path):
    """
    Dynamically load CNN_LSTM_Forecaster class from a Python file.
    
    Args:
        model_file_path: Path to the Python file containing the model class
        
    Returns:
        The CNN_LSTM_Forecaster class
    """
    spec = importlib.util.spec_from_file_location("cnn_lstm_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cnn_lstm_module"] = module
    spec.loader.exec_module(module)
    return module.CNN_LSTM_Forecaster


def load_model(model_config, input_dim, device='cpu', print_debug=True):
    """
    Load a model from configuration.
    
    Args:
        model_config: Dictionary with model configuration
        input_dim: Number of input features
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    print(CYAN + "Loading " + model_config['name'] + "..." + RESET)
    
    if print_debug:
        print("     Model file: " + model_config['model_file'])
        print("     Weights: " + model_config['path'])
    
    # Load model class dynamically
    CNN_LSTM_Forecaster = load_model_class(model_config['model_file'])
    
    # Create model instance
    model = CNN_LSTM_Forecaster(
        input_dim=input_dim,
        seq_len=model_config['seq_len'],
        output_dim=model_config['horizon']
    ).to(device)
    
    # Load weights
    if not os.path.exists(model_config['path']):
        print(RED + "Model weights not found: " + model_config['path'] + RESET)
        return None
    
    model.load_state_dict(torch.load(model_config['path'], map_location=device, weights_only=True))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(GREEN + "Model loaded: " + format(total_params, ",") + " parameters" + RESET + "\n")
    
    return model


def generate_continuous_predictions(model, dataloader, device, output_horizon, seq_len, scaler, df_test=None, start_date=None, end_date=None):
    """
    Generate continuous predictions by sampling every output_horizon-th sample.
    This creates a non-overlapping timeline similar to plot_full_test_set_predictions.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device to run on
        output_horizon: Forecast horizon
        seq_len: Sequence length
        scaler: Scaler for denormalization
        df_test: Test dataframe with timestamps (optional, needed for date filtering)
        start_date: Optional start date (str format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM') to filter predictions
        end_date: Optional end date (str format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM') to filter predictions
        
    Returns:
        pred_timeline: Continuous prediction timeline
        target_timeline: Continuous ground truth timeline
        pred_time_indices: Indices for predictions relative to test set
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
            all_inputs.append(X_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # --- Compute errors over ALL (overlapping) predictions ---
    # Flatten all predictions/targets (all sliding-window predictions)
    all_preds_flat = all_preds.flatten()
    all_targets_flat = all_targets.flatten()

    # Denormalize if scaler provided (Load is last feature)
    if scaler is not None:
        n_features = scaler.data_min_.shape[0]

        pred_dummy = np.zeros((len(all_preds_flat), n_features))
        targ_dummy = np.zeros((len(all_targets_flat), n_features))
        pred_dummy[:, -1] = all_preds_flat
        targ_dummy[:, -1] = all_targets_flat

        all_preds_flat_den = scaler.inverse_transform(pred_dummy)[:, -1]
        all_targets_flat_den = scaler.inverse_transform(targ_dummy)[:, -1]
    else:
        all_preds_flat_den = all_preds_flat
        all_targets_flat_den = all_targets_flat

    # Safe MAPE calculation
    eps = 1e-8
    mse_all = np.mean((all_preds_flat_den - all_targets_flat_den) ** 2)
    mae_all = np.mean(np.abs(all_preds_flat_den - all_targets_flat_den))
    rmse_all = np.sqrt(mse_all)
    mape_all = np.mean(np.abs((all_targets_flat_den - all_preds_flat_den) / (all_targets_flat_den + eps))) * 100

    all_metrics = {
        'MSE_all': mse_all,
        'MAE_all': mae_all,
        'RMSE_all': rmse_all,
        'MAPE_all': mape_all
    }
    
    # Sample every output_horizon-th prediction to avoid overlaps
    # If start_date is provided, find the starting index
    start_idx = 0
    end_idx = len(all_preds)
    
    if df_test is not None and start_date is not None:
        import pandas as pd
        
        # Get timestamps for all predictions
        timestamps = df_test.index[:]
        start_dt = pd.Timestamp(start_date)
        
        # Find first prediction that starts at or after start_date
        # Each prediction i corresponds to timestamps [seq_len + i : seq_len + i + output_horizon]
        for i in range(0, len(all_preds)):
            pred_start_time_idx = seq_len + i
            if pred_start_time_idx < len(timestamps):
                pred_start_time = timestamps[pred_start_time_idx]
                if pred_start_time >= start_dt:
                    start_idx = i
                    print(CYAN + f"   Starting predictions at index {start_idx}, time {pred_start_time}" + RESET)
                    break
    
    if df_test is not None and end_date is not None:
        import pandas as pd
        
        # Get timestamps for all predictions
        timestamps = df_test.index[:]
        end_dt = pd.Timestamp(end_date)
        
        # Find last prediction that ends at or before end_date
        for i in range(start_idx, len(all_preds), output_horizon):
            pred_end_time_idx = seq_len + i + output_horizon - 1
            if pred_end_time_idx < len(timestamps):
                pred_end_time = timestamps[pred_end_time_idx]
                if pred_end_time <= end_dt:
                    end_idx = i + output_horizon
                else:
                    break
        
        print(CYAN + f"   Ending predictions at index {end_idx}" + RESET)
    
    selected_indices = np.arange(start_idx, end_idx, output_horizon)
    
    selected_preds = all_preds[selected_indices]
    selected_targets = all_targets[selected_indices]
    selected_inputs = all_inputs[selected_indices]
    
    # Flatten predictions and targets
    pred_timeline = selected_preds.flatten()
    target_timeline = selected_targets.flatten()
    
    # Extract input timeline (Load is last feature)
    input_timeline_list = []
    input_timeline_list.append(selected_inputs[0, :, -1])
    
    for i in range(1, len(selected_inputs)):
        input_timeline_list.append(selected_inputs[i, -output_horizon:, -1])
    
    input_timeline = np.concatenate(input_timeline_list)
    
    # Denormalize
    if scaler is not None:
        n_features = scaler.data_min_.shape[0]
        
        # Denormalize input
        input_dummy = np.zeros((len(input_timeline), n_features))
        input_dummy[:, -1] = input_timeline
        input_timeline = scaler.inverse_transform(input_dummy)[:, -1]
        
        # Denormalize predictions
        pred_dummy = np.zeros((len(pred_timeline), n_features))
        pred_dummy[:, -1] = pred_timeline
        pred_timeline = scaler.inverse_transform(pred_dummy)[:, -1]
        
        # Denormalize targets
        target_dummy = np.zeros((len(target_timeline), n_features))
        target_dummy[:, -1] = target_timeline
        target_timeline = scaler.inverse_transform(target_dummy)[:, -1]
    
    # Calculate time indices for predictions
    pred_time_indices = []
    for i, sample_idx in enumerate(selected_indices):
        start_time_idx = seq_len + sample_idx
        end_time_idx = start_time_idx + output_horizon
        pred_time_indices.extend(range(start_time_idx, end_time_idx))
    
    pred_time_indices = pred_time_indices[:len(pred_timeline)]
    
    if df_test is not None and (start_date is not None or end_date is not None):
        print(CYAN + f"   Generated predictions from {df_test.index[pred_time_indices[0]]} to {df_test.index[pred_time_indices[-1]]}" + RESET)
        print(f"   Total timesteps: {len(pred_timeline)} prediction timesteps")
    
    return pred_timeline, target_timeline, pred_time_indices, len(selected_preds), all_metrics


def plot_model_comparison(models_data, df_test, plot_integral_difference=False):
    """
    Plot predictions from multiple models in one figure.
    Each model's predictions are shown as continuous timelines (non-overlapping).
    
    Args:
        models_data: Dict with model predictions {model_name: (pred_timeline, target_timeline, pred_time_indices, config, n_samples)}
        df_test: Test dataframe with timestamps
        plot_integral_difference: Whether to plot cumulative error
    """
    print(CYAN + "Creating comparison plot..." + RESET)
    
    # Calculate and print overall test-set statistics (average and peak consumption)
    if 'Load' in df_test.columns:
        avg_test_load = df_test['Load'].mean()
        peak_test_load = df_test['Load'].max()
        print(f" Test set: Average load = {avg_test_load:.2f} W, Peak load = {peak_test_load:.2f} W")
    else:
        print(YELLOW + " Test set does not contain 'Load' column; skipping avg/peak stats." + RESET)

    # Create figure
    if plot_integral_difference:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), 
                                        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 6))
    
    # Get timestamps for x-axis
    timestamps = df_test.index[:]
    
    # Plot ground truth (only once, using the first model's targets)
    ground_truth_plotted = False
    
    for model_name, (pred_timeline, target_timeline, pred_time_indices, config, n_samples, all_metrics) in models_data.items():
        color = config['color']
        label = config['name']
        horizon = config['horizon']
        
        # Get actual timestamps for predictions
        pred_timestamps = timestamps[pred_time_indices]
        
        # Plot ground truth only once (use first model's target timeline)
        if not ground_truth_plotted:
            ax1.plot(pred_timestamps, target_timeline, 
                    color='blue', linewidth=1.8, alpha=0.8, 
                    label='Ground Truth', zorder=1)
            ground_truth_plotted = True
        
        # Plot predictions for this model
        ax1.plot(pred_timestamps, pred_timeline, 
                color=color, linewidth=1.8, alpha=0.8, 
                label=label, zorder=2)
        
        # Calculate metrics
        mse = np.mean((pred_timeline - target_timeline) ** 2)
        mae = np.mean(np.abs(pred_timeline - target_timeline))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(pred_timeline - target_timeline) / (target_timeline + 1e-8)) * 100

        # R^2 (coefficient of determination): 1 - SS_res / SS_tot
        ss_res = np.sum((target_timeline - pred_timeline) ** 2)
        ss_tot = np.sum((target_timeline - np.mean(target_timeline)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        print(
            "   " + label + ": MSE = " + "{:.2f}".format(mse)
            + " W², MAE = " + "{:.2f}".format(mae)
            + " W, RMSE = " + "{:.2f}".format(rmse)
            + " W, MAPE = " + "{:.2f}".format(mape)
            + " %, R2 = " + "{:.3f}".format(r2)
            + " (" + str(n_samples) + " samples)"
        )
        # Print overlapping (ALL) errors if available
        if all_metrics is not None:
            try:
                print(f"   Overlapping (ALL) preds: MSE_all = {all_metrics['MSE_all']:.2f} W², MAE_all = {all_metrics['MAE_all']:.2f} W, RMSE_all = {all_metrics['RMSE_all']:.2f} W, MAPE_all = {all_metrics['MAPE_all']:.2f} %")
            except Exception:
                pass
        
        # Plot integral difference if requested
        if plot_integral_difference:
            error = pred_timeline - target_timeline
            cumulative_error = np.cumsum(error)
            ax2.plot(pred_timestamps, cumulative_error, 
                    color=color, linewidth=1.0, alpha=0.8, 
                    label=f'{label}')
    
    # Configure main plot
    ax1.set_ylabel('Electrical Load [W]', fontsize=15)
    ax1.set_title('Load Prediction Sequence over 10 Days - House 2 without the Heat Pump', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=15)
    #ax1.grid(True, alpha=0.3)
    
    # Format x-axis with date formatter
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    
    # Configure integral difference plot
    if plot_integral_difference:
        ax2.set_xlabel('Time', fontsize=13)
        ax2.set_ylabel('Cumulative Error [W]', fontsize=13)
        ax2.set_title('Cumulative Prediction Error', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m. %H:%M'))
        ax2.tick_params(axis='x', labelsize=13)
        ax2.tick_params(axis='y', labelsize=13)
    else:
        ax1.set_xlabel('Time', fontsize=15)
    
    plt.tight_layout()
    
    # Save plot with timestamp
    try:
        first_timestamp = timestamps[0]
        timestamp_str = first_timestamp.strftime('%Y-%m-%d_%H-%M')
    except Exception:
        timestamp_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    out_fname = f'model_comparison_{timestamp_str}.png'
    plt.savefig(out_fname, dpi=300, bbox_inches='tight')
    
    plt.show(block=False)
    plt.pause(0.1)
    
    print(GREEN + f"Comparison plot created and saved to {out_fname}." + RESET)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple CNN-LSTM models')
    parser.add_argument('--data_path', type=str,
                        default='data/data/dfA_300s.hdf',
                        help='Path to HDF5 data file')
    parser.add_argument('--weather_csv_path', type=str,
                        default='data/data/weather_data_house_a_LUZ.csv',
                        help='Path to weather CSV file (set to "none" to disable)')
    #----------------------------------selected models House A----------------------------------
    # parser.add_argument('--models', type=str, nargs='+',
    #                     default=['hor_48_A_24_5'],
    #                     choices=['hor_48_A_24_5'],
    #                     help='Models to compare')


    #----------------------------------selected models House E----------------------------------
    # parser.add_argument('--models', type=str, nargs='+',
    #                     default=['hor_16_E_ker5', 'hor_24_E', 'hor_32_E_24_2', 'hor_48_E_24_2'],
    #                     choices=['hor_16_E_ker5', 'hor_24_E', 'hor_32_E_24_2', 'hor_48_E_24_2'],
    #                     help='Models to compare')
    
    # parser.add_argument('--models', type=str, nargs='+',
    #                     default=['hor_48_E_24_2'],
    #                     choices=['hor_48_E_24_2'],
    #                     help='Models to compare')
    
    
    #----------------------------------selected models house A no HP----------------------------------
    # parser.add_argument('--models', type=str, nargs='+',
    #                 default=['hor_16_A_no_HP_ker5', 'hor_24_A_no_HP', 'hor_32_A_no_HP_24_1', 'hor_48_A_no_HP_24_2'],
    #                 choices=['hor_16_A_no_HP_ker5', 'hor_24_A_no_HP', 'hor_32_A_no_HP_24_1', 'hor_48_A_no_HP_24_2'],
    #                 help='Models to compare')
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=['hor_48_A_no_HP_24_2'],
                        choices=['hor_48_A_no_HP_24_2'],
                        help='Models to compare')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--plot_integrated_difference', action='store_true',
                        help='Plot cumulative integral difference')
    parser.add_argument('--use_cyclic_encoding', action='store_true', default=True,
                        help='Use cyclic encoding for temporal features')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for predictions (format: YYYY-MM-DD or YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for predictions (format: YYYY-MM-DD or YYYY-MM-DD HH:MM)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(BRIGHT_BLUE + ('=' * 60) + RESET)
    print(BRIGHT_BLUE + "  CNN-LSTM Model Comparison" + RESET)
    print(BRIGHT_BLUE + ('=' * 60) + RESET + "\n")
    print("     Using device: " + str(device))
    print("     Models to compare: " + ', '.join(args.models) + "\n")

    # Handle weather path
    weather_csv_path = None if args.weather_csv_path.lower() == 'none' else args.weather_csv_path
    
    # Load data
    
    
    if HOUSE_TYPE == 'A':
        args.data_path = 'data/data/dfA_300s.hdf'
        print(CYAN + "Loading data from " + args.data_path + "..." + RESET)
        df = load_energy_hdf_to_pandas(
            args.data_path,
            weather_csv_path=weather_csv_path,
            plot_data=False,
            use_cyclic_encoding=args.use_cyclic_encoding, 
            print_debug=PRINT_DEBUG
        )
    elif HOUSE_TYPE == 'E':
        args.data_path = 'data/data/dfE_300s.hdf'
        print(CYAN + "Loading data from " + args.data_path + "..." + RESET)
        df = load_energy_hdf_to_pandas_2nd_house(
            args.data_path,
            weather_csv_path=weather_csv_path,
            plot_data=False,
            use_cyclic_encoding=args.use_cyclic_encoding, 
            print_debug=PRINT_DEBUG
        )
    else:
        raise ValueError("Invalid HOUSE_TYPE value. Choose 'A' or 'E'.")
    
    has_temperature = 'Temperature' in df.columns
    
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15, print_debug=PRINT_DEBUG)
    
    max_horizon = max([MODELS[m]['horizon'] for m in args.models])
    max_seq_len = max([MODELS[m]['seq_len'] for m in args.models])
    
    train_set = EnergyDataset(df_train, max_seq_len, max_horizon, 
                              normalize=True, scaler=None, 
                              use_cyclic_encoding=args.use_cyclic_encoding)
    test_set = EnergyDataset(df_test, max_seq_len, max_horizon, 
                            normalize=True, scaler=train_set.scaler,
                            use_cyclic_encoding=args.use_cyclic_encoding)
    
    print(GREEN + "Data loaded and datasets created." + RESET)
    
    input_dim = train_set.X.shape[2]
    
    if PRINT_DEBUG:
        print("   Input features: " + str(input_dim))
        print("   Features: " + str(train_set.model_feature_cols) + "\n")
    
    # Load models and generate predictions
    models_data = {}
    
    for model_key in args.models:
        config = MODELS[model_key]
        
        # For models without temperature, we need to prepare data without temperature
        if not config['use_temperature'] and has_temperature:
            print(YELLOW + "Note: " + config['name'] + " was trained WITHOUT temperature data" + RESET)
            # Load data without temperature
            df_no_temp = load_energy_hdf_to_pandas(
                args.data_path,
                weather_csv_path=None,  # No temperature
                plot_data=False,
                use_cyclic_encoding=args.use_cyclic_encoding
            )
            df_train_no_temp, df_val_no_temp, df_test_no_temp = split_dataframe(df_no_temp, 0.7, 0.15, 0.15)
            
            # Create dataset without temperature
            train_set_no_temp = EnergyDataset(df_train_no_temp, config['seq_len'], config['horizon'],
                                              normalize=True, scaler=None,
                                              use_cyclic_encoding=args.use_cyclic_encoding)
            test_set_model = EnergyDataset(df_test_no_temp, config['seq_len'], config['horizon'],
                                           normalize=True, scaler=train_set_no_temp.scaler,
                                           use_cyclic_encoding=args.use_cyclic_encoding)
            
            model_input_dim = test_set_model.X.shape[2]
            model_scaler = train_set_no_temp.scaler
        else:
            # Use normal dataset with temperature
            test_set_model = EnergyDataset(df_test, config['seq_len'], config['horizon'],
                                           normalize=True, scaler=train_set.scaler,
                                           use_cyclic_encoding=args.use_cyclic_encoding)
            model_input_dim = input_dim
            model_scaler = train_set.scaler
        
        # Load model
        model = load_model(config, model_input_dim, device)
        if model is None:
            print(YELLOW + "    Skipping " + config['name'] + RESET + "\n")
            continue
        
        # Create dataloader
        test_loader = DataLoader(test_set_model, batch_size=args.batch_size)
        
        # Generate predictions
        print(CYAN + "Generating predictions for " + config['name'] + "..." + RESET)
        pred_timeline, target_timeline, pred_time_indices, n_samples, all_metrics = generate_continuous_predictions(
            model, test_loader, device, 
            config['horizon'], config['seq_len'],
            model_scaler, df_test, args.start_date, args.end_date
        )

        # Print errors computed over ALL (overlapping) predictions for this model
        print(CYAN + "Errors over ALL predictions for " + config['name'] + ":" + RESET)
        print(f"   MSE_all: {all_metrics['MSE_all']:.3f} W^2 | MAE_all: {all_metrics['MAE_all']:.3f} W | RMSE_all: {all_metrics['RMSE_all']:.3f} W | MAPE_all: {all_metrics['MAPE_all']:.2f}%")
        
        models_data[model_key] = (pred_timeline, target_timeline, pred_time_indices, config, n_samples, all_metrics)
        print(GREEN + "Generated " + str(n_samples) + " non-overlapping predictions (" + str(len(pred_timeline)) + " timesteps)" + RESET + "\n")
    
    if not models_data:
        print(RED + "No models loaded successfully" + RESET)
        return
    
    # Plot comparison
    plot_model_comparison(
        models_data, df_test,
        plot_integral_difference=args.plot_integrated_difference
    )
    
    # Final summary: print only the overlapping (ALL) errors per model
    print('\n' + BRIGHT_BLUE + 'Final summary (Overlapping / ALL predictions):' + RESET)
    for model_key, model_vals in models_data.items():
        # model_vals = (pred_timeline, target_timeline, pred_time_indices, config, n_samples, all_metrics)
        try:
            config = model_vals[3]
            all_metrics = model_vals[5]
            name = config.get('name', model_key)
            if all_metrics is not None:
                print(f"  {name}: MSE_all={all_metrics['MSE_all']:.2f} W^2 | MAE_all={all_metrics['MAE_all']:.2f} W | RMSE_all={all_metrics['RMSE_all']:.2f} W | MAPE_all={all_metrics['MAPE_all']:.2f}%")
            else:
                print(f"  {name}: Overlapping metrics not available")
        except Exception:
            print(f"  {model_key}: Error reading overlapping metrics")

    # Wait for user input
    input("\n" + YELLOW + "Press Enter to exit..." + RESET)


if __name__ == "__main__":
    main()
