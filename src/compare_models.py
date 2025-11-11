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
from torch.utils.data import DataLoader
from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe

# ANSI Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BRIGHT_BLUE = '\033[94;1m'
CYAN = '\033[96m'
RESET = '\033[0m'

PRINT_DEBUG = False

# Model configurations
MODELS = {
    'hor_16': {
        'name': '16-step (no temp)',
        'path': 'NN_storage_good_model/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'NN_storage_good_model/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': False,
        'color': 'black'
    },
    'hor_16_temp': {
        'name': '16-step',
        'path': 'NN_storage_hor_16/NN_storage_3/NN_weights/cnn_lstm_forecaster.pth',
        'model_file': 'NN_storage_hor_16/NN_storage_3/cnn_lstm.py',
        'horizon': 16,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'orange'
    },
    'hor_24': {
        'name': '24-step',
        'path': 'NN_storage_hor_24/NN_storage_5/NN_weights/best_model.pth',
        'model_file': 'NN_storage_hor_24/NN_storage_5/cnn_lstm.py',
        'horizon': 24,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'green'
    },
    'hor_32': {
        'name': '32-step',
        'path': 'NN_storage_hor_32/NN_storage_2/NN_weights/best_model.pth',
        'model_file': 'NN_storage_hor_32/NN_storage_2/cnn_lstm.py',
        'horizon': 32,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'black'
    },
    'hor_48': {
        'name': '48-step',
        'path': 'NN_storage_hor_48/NN_storage/NN_weights/best_model.pth',
        'model_file': 'NN_storage_hor_48/NN_storage/cnn_lstm.py',
        'horizon': 48,
        'seq_len': 192,
        'use_temperature': True,
        'color': 'red'
    }
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


def generate_continuous_predictions(model, dataloader, device, output_horizon, seq_len, scaler):
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
    
    # Sample every output_horizon-th prediction to avoid overlaps
    selected_indices = np.arange(0, len(all_preds), output_horizon)
    
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
    
    return pred_timeline, target_timeline, pred_time_indices, len(selected_preds)


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
    
    # Create figure
    if plot_integral_difference:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 14), 
                                        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(24, 10))
    
    # Get timestamps for x-axis
    timestamps = df_test.index[:]
    
    # Plot ground truth (only once, using the first model's targets)
    ground_truth_plotted = False
    
    for model_name, (pred_timeline, target_timeline, pred_time_indices, config, n_samples) in models_data.items():
        color = config['color']
        label = config['name']
        horizon = config['horizon']
        
        # Get actual timestamps for predictions
        pred_timestamps = timestamps[pred_time_indices]
        
        # Plot ground truth only once (use first model's target timeline)
        if not ground_truth_plotted:
            ax1.plot(pred_timestamps, target_timeline, 
                    color='blue', linewidth=1.2, alpha=0.7, 
                    label='Ground Truth', zorder=1)
            ground_truth_plotted = True
        
        # Plot predictions for this model
        ax1.plot(pred_timestamps, pred_timeline, 
                color=color, linewidth=1.2, alpha=0.8, 
                label=f'{label} Prediction', zorder=2)
        
        # Calculate metrics
        mse = np.mean((pred_timeline - target_timeline) ** 2)
        mae = np.mean(np.abs(pred_timeline - target_timeline))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(pred_timeline - target_timeline)/(target_timeline + 1e-8)) * 100  
        print("   " + label + ": MSE = " + "{:.2f}".format(mse) + " W², MAE = " + "{:.2f}".format(mae) + " W, RMSE = " + "{:.2f}".format(rmse) + " W, MAPE = " + "{:.2f}".format(mape) + " % (" + str(n_samples) + " samples)")
        
        # Plot integral difference if requested
        if plot_integral_difference:
            error = pred_timeline - target_timeline
            cumulative_error = np.cumsum(error)
            ax2.plot(pred_timestamps, cumulative_error, 
                    color=color, linewidth=1.0, alpha=0.8, 
                    label=f'{label}')
    
    # Configure main plot
    ax1.set_ylabel('Load [W]', fontsize=12)
    ax1.set_title('Model Comparison - Non-Overlapping Predictions on Full Test Set', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Configure integral difference plot
    if plot_integral_difference:
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Cumulative Error [W]', fontsize=12)
        ax2.set_title('Cumulative Prediction Error', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax1.set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    print(GREEN + "Comparison plot created." + RESET)


def main():
    parser = argparse.ArgumentParser(description='Compare multiple CNN-LSTM models')
    parser.add_argument('--data_path', type=str,
                        default='data/data/dfA_300s.hdf',
                        help='Path to HDF5 data file')
    parser.add_argument('--weather_csv_path', type=str,
                        default='data/data/weather_data_house_a_LUZ.csv',
                        help='Path to weather CSV file (set to "none" to disable)')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['hor_16_temp', 'hor_24', 'hor_32', 'hor_48'],
                        choices=['hor_16_temp', 'hor_24', 'hor_32', 'hor_48'],
                        help='Models to compare')
    # parser.add_argument('--models', type=str, nargs='+',
    #                 default=['hor_16', 'hor_16_temp'],
    #                 choices=['hor_16', 'hor_16_temp'],
    #                 help='Models to compare')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--plot_integrated_difference', action='store_true',
                        help='Plot cumulative integral difference')
    parser.add_argument('--use_cyclic_encoding', action='store_true', default=True,
                        help='Use cyclic encoding for temporal features')
    
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
    print(CYAN + "Loading data from " + args.data_path + "..." + RESET)
    df = load_energy_hdf_to_pandas(
        args.data_path,
        weather_csv_path=weather_csv_path,
        plot_data=False,
        use_cyclic_encoding=args.use_cyclic_encoding, 
        print_debug=PRINT_DEBUG
    )
    
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
        pred_timeline, target_timeline, pred_time_indices, n_samples = generate_continuous_predictions(
            model, test_loader, device, 
            config['horizon'], config['seq_len'],
            model_scaler
        )
        
        models_data[model_key] = (pred_timeline, target_timeline, pred_time_indices, config, n_samples)
        print(GREEN + "Generated " + str(n_samples) + " non-overlapping predictions (" + str(len(pred_timeline)) + " timesteps)" + RESET + "\n")
    
    if not models_data:
        print(RED + "No models loaded successfully" + RESET)
        return
    
    # Plot comparison
    plot_model_comparison(
        models_data, df_test,
        plot_integral_difference=args.plot_integrated_difference
    )
    
    # Wait for user input
    input("\n" + YELLOW + "Press Enter to exit..." + RESET)


if __name__ == "__main__":
    main()
