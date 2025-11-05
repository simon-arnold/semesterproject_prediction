"""
Evaluate a saved model on the test dataset.

This script loads a previously trained model and evaluates it on test data.
Can be run independently from training.

Example usage:
    evaluate_saved_model.py --plot_train_val_test --plot_full --predict_at_date 2018-08-23 --n_examples 5
"""



from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe
from models.cnn_lstm_forecaster import CNN_LSTM_Forecaster
from training.evaluate import evaluate_model, plot_full_test_set_predictions, plot_raw_dataframe, plot_multiple_predictions_at_date
from torch.utils.data import DataLoader
import torch
import argparse
import os

# ANSI Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'

# ========================================
# CONFIGURATION: Must match training setup
# ========================================
# Set this to match the encoding used when training the model!
USE_CYCLIC_ENCODING = True  # True: 8 features (cyclic), False: 6 features (raw)


def load_test_data(data_path="data/data/dfA_300s.hdf", weather_csv_path="data/data/weather_data_house_a_LUZ.csv", seq_len=192, output_horizon=16):
    """
    Load and prepare test dataset with proper scaler from training data.
    
    Args:
        data_path: Path to energy HDF5 file
        weather_csv_path: Path to weather CSV file (optional, set to None to disable)
        seq_len: Sequence length for input
        output_horizon: Number of timesteps to predict
    
    Returns:
        test_loader: DataLoader for test set
        scaler: The scaler fitted on training data
        train_set: Training dataset (for scaler access)
        test_set: Test dataset
        df_train: Raw train dataframe
        df_val: Raw validation dataframe
        df_test: Raw test dataframe
        input_dim: Number of input features (auto-detected)
    """
    print(f"{CYAN}📊 Loading data from {data_path}...{RESET}")
    
    # Load energy data with optional weather data
    df = load_energy_hdf_to_pandas(
        data_path, 
        weather_csv_path=weather_csv_path,
        plot_data=False, 
        use_cyclic_encoding=USE_CYCLIC_ENCODING
    )
    
    # Detect if Temperature column exists
    has_temperature = 'Temperature' in df.columns
    
    if has_temperature:
        print(f"{GREEN}🌡️  Temperature data found and will be used!{RESET}")
    else:
        print(f"{YELLOW}⚠️  No Temperature data found, using only temporal features.{RESET}")
    
    # Split (same splits as training)
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15)
    
    # Create datasets with scaler from training data
    train_set = EnergyDataset(df_train, seq_len, output_horizon, normalize=True, scaler=None, use_cyclic_encoding=USE_CYCLIC_ENCODING)
    test_set = EnergyDataset(df_test, seq_len, output_horizon, normalize=True, scaler=train_set.scaler, use_cyclic_encoding=USE_CYCLIC_ENCODING)
    
    # Auto-detect input_dim from dataset
    input_dim = train_set.X.shape[2]  # Shape: (n_samples, seq_len, n_features)
    
    print(f"{CYAN}📋 Data Configuration:{RESET}")
    print(f"   Feature encoding: {'CYCLIC' if USE_CYCLIC_ENCODING else 'RAW'}")
    print(f"   Temperature: {'✓ Included' if has_temperature else '✗ Not included'}")
    print(f"   Input features (input_dim): {input_dim}")
    print(f"   Features: {train_set.model_feature_cols}")
    
    print(f"\n{CYAN}📊 Scaler info (fitted on training data):{RESET}")
    print(f"   Min: {train_set.scaler.data_min_}")
    print(f"   Max: {train_set.scaler.data_max_}")
    print(f"   Train samples: {len(df_train)}")
    print(f"   Val samples: {len(df_val)}")
    print(f"   Test samples: {len(df_test)}")
    
    # Create DataLoader
    test_loader = DataLoader(test_set, batch_size=64)
    
    print(f"{GREEN}✅ Data loaded successfully!{RESET}\n")

    return test_loader, train_set.scaler, train_set, test_set, df_train, df_val, df_test, input_dim


def load_model(model_path, input_dim, seq_len=192, output_horizon=16, device='cpu'):
    """
    Load a saved model from .pth file.
    
    Args:
        model_path: Path to .pth file
        input_dim: Number of input features (auto-detected from data)
        seq_len: Sequence length (must match training)
        output_horizon: Output horizon (must match training)
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
    """
    print(f"{CYAN}🔄 Loading model from {model_path}...{RESET}")
    
    model = CNN_LSTM_Forecaster(
        input_dim=input_dim, 
        seq_len=seq_len, 
        output_dim=output_horizon
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{GREEN}✅ Model loaded successfully!{RESET}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Input features: {input_dim}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Output horizon: {output_horizon}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved CNN-LSTM model')
    parser.add_argument('--model_path', type=str, 
                        default='NN_storage/NN_weights/cnn_lstm_forecaster.pth',
                        help='Path to saved model (.pth file)')
    parser.add_argument('--data_path', type=str,
                        default='data/data/dfA_300s.hdf',
                        help='Path to HDF5 data file')
    parser.add_argument('--weather_csv_path', type=str,
                        default='data/data/weather_data_house_a_LUZ.csv',
                        help='Path to weather CSV file (set to "none" to disable)')
    parser.add_argument('--seq_len', type=int, default=192,
                        help='Sequence length (must match training)')
    parser.add_argument('--output_horizon', type=int, default=16,
                        help='Output horizon (must match training)')
    parser.add_argument('--n_examples', type=int, default=1,
                        help='Number of prediction examples to plot')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--plot_full', action='store_true', 
                        help='Plot entire test set in one continuous timeline')
    parser.add_argument('--skip_examples', action='store_true',
                        help='Skip individual example plots (useful with --plot_full)')
    parser.add_argument('--plot_train_val_test', action='store_true',
                        help='Plot only test data without predictions (no model loading needed)')
    parser.add_argument('--predict_at_date', type=str, default=None,
                        help='Make prediction starting at specific date (format: YYYY-MM-DD, e.g., 2018-08-31)')
    parser.add_argument('--plot_integrated_difference', action='store_true',
                        help='Plot cumulative integral difference in full test set plot', default=False)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  CNN-LSTM Load Forecasting Model Evaluation{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    print(f"  Using device: {device}")
    
    # Display encoding mode
    if USE_CYCLIC_ENCODING:
        print(f"  {YELLOW}🔄 Feature Encoding: CYCLIC{RESET}")
    else:
        print(f"  {YELLOW}📊 Feature Encoding: RAW{RESET}")
    
    print(f"{BLUE}{'='*60}{RESET}\n")

    # Handle weather path (allow "none" to disable)
    weather_csv_path = None if args.weather_csv_path.lower() == 'none' else args.weather_csv_path
    
    test_loader, scaler, train_set, test_set, df_train, df_val, df_test, input_dim = load_test_data(
        args.data_path,
        weather_csv_path,
        args.seq_len,
        args.output_horizon
    )
    
    if not os.path.exists(args.model_path):
        print(f"{RED}❌ Error: Model not found at {args.model_path}{RESET}")
        print(f"{RED}   Please train a model first or specify correct path with --model_path{RESET}")
        return
    
    model = load_model(
        args.model_path,
        input_dim,  # Auto-detected from data
        args.seq_len,
        args.output_horizon,
        device
    )
    
    
    # Plotting specific cases:
    
    # Plotting only data loader data
    if args.plot_train_val_test:
        plot_raw_dataframe((df_train, df_val, df_test), title="Complete Dataset")
    

    # Prediction at specific date
    # Possibly multiple predictions spaced 12 hours apart
    if not args.skip_examples:
        plot_multiple_predictions_at_date(
            model,
            test_set=test_set,
            start_date=args.predict_at_date,
            n_examples=args.n_examples,
            seq_len=args.seq_len,
            output_horizon=args.output_horizon,
            device=device
        )
        
        
        
    # Plot predictions all 16 timesteps - get an overview of full prediction
    if args.plot_full:
        all_preds, all_targets = plot_full_test_set_predictions(
            model, test_loader, device, 
            output_horizon=args.output_horizon,
            df_test=df_test,
            seq_len=args.seq_len,
            scaler=train_set.scaler, 
            plot_integral_difference=args.plot_integrated_difference, 
            integral_reset_interval=6
        )
        
        
    # 3. Evaluate
    test_metrics = evaluate_model(model, test_loader, device)

    
    # Wait for user input before closing
    input(f"\n{YELLOW}Press Enter to exit...{RESET}")



if __name__ == "__main__":
    main()
