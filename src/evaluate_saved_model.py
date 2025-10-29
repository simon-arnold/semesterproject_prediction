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
USE_CYCLIC_ENCODING = False  # True: 8 features (cyclic), False: 6 features (raw)


def load_test_data(data_path="data/data/dfA_300s.hdf", seq_len=192, output_horizon=16):
    """
    Load and prepare test dataset with proper scaler from training data.
    
    Returns:
        test_loader: DataLoader for test set
        scaler: The scaler fitted on training data
        df_train: Raw train dataframe
        df_val: Raw validation dataframe
        df_test: Raw test dataframe
    """
    print(f"{CYAN} Loading data from {data_path}...{RESET}")
    df = load_energy_hdf_to_pandas(data_path, plot_data=False, use_cyclic_encoding=USE_CYCLIC_ENCODING)
    
    # Split (same splits as training)
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15)
    
    # Create datasets with scaler from training data
    train_set = EnergyDataset(df_train, seq_len, output_horizon, normalize=True, scaler=None, use_cyclic_encoding=USE_CYCLIC_ENCODING)
    test_set = EnergyDataset(df_test, seq_len, output_horizon, normalize=True, scaler=train_set.scaler, use_cyclic_encoding=USE_CYCLIC_ENCODING)
    
    print(f"-- Scaler info (fitted on training data): --")
    print(f"   Min: {train_set.scaler.data_min_}")
    print(f"   Max: {train_set.scaler.data_max_}")
    print(f"   Train samples: {len(df_train)}")
    print(f"   Val samples: {len(df_val)}")
    print(f"   Test samples: {len(df_test)}")
    
    # Create DataLoader
    test_loader = DataLoader(test_set, batch_size=64)
    
    print(f"{GREEN} Data loaded successfully!{RESET}")

    return test_loader, train_set.scaler, train_set, test_set, df_train, df_val, df_test


def load_model(model_path, seq_len=192, output_horizon=16, device='cpu'):
    """
    Load a saved model from .pth file.
    
    Args:
        model_path: Path to .pth file
        seq_len: Sequence length (must match training)
        output_horizon: Output horizon (must match training)
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
    """
    print(f"{CYAN} Loading model from {model_path}...{RESET}")
    
    # Initialize model architecture
    # IMPORTANT: input_dim must match the model's training configuration!
    # Use 8 for cyclic encoding (Year + 6 cyclic + Load)
    # Use 6 for raw features (Year, Month, Day, Weekday, Timestep, Load)
    input_dim = 8 if USE_CYCLIC_ENCODING else 6
    
    model = CNN_LSTM_Forecaster(
        input_dim=input_dim, 
        seq_len=seq_len, 
        output_dim=output_horizon
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"{GREEN} Model loaded successfully!{RESET}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved CNN-LSTM model')
    parser.add_argument('--model_path', type=str, 
                        default='NN_storage/NN_weights/cnn_lstm_forecaster.pth',
                        help='Path to saved model (.pth file)')
    parser.add_argument('--data_path', type=str,
                        default='data/data/dfA_300s.hdf',
                        help='Path to HDF5 data file')
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
    parser.add_argument('--predict_at_date', type=str, default=None, #Enter the date&time when the prediction should start
                        help='Make prediction starting at specific date (format: YYYY-MM-DD, e.g., 2018-08-31)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}\n")

    test_loader, scaler, train_set, test_set, df_train, df_val, df_test = load_test_data(
        args.data_path,
        args.seq_len,
        args.output_horizon
    )
    
    if not os.path.exists(args.model_path):
        print(f"{RED}  Error: Model not found at {args.model_path}{RESET}")
        print(f"{RED}   Please train a model first or specify correct path with --model_path{RESET}")
        return
    
    model = load_model(
        args.model_path,
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
            scaler=train_set.scaler
        )
        
        
    # 3. Evaluate
    test_metrics = evaluate_model(model, test_loader, device)

    
    # Wait for user input before closing
    input(f"\n{YELLOW}Press Enter to exit...{RESET}")



if __name__ == "__main__":
    main()
