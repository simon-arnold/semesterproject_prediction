"""
Evaluate a saved model on the test dataset.

This script loads a previously trained model and evaluates it on test data.
Can be run independently from training.

Usage:
    python src/evaluate_saved_model.py
    python src/evaluate_saved_model.py --model_path NN_storage/NN_weights/custom_model.pth
"""

from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe
from models.cnn_lstm_forecaster import CNN_LSTM_Forecaster
from training.evaluate import evaluate_model, plot_predictions, plot_full_test_set, plot_full_test_set_non_overlapping, plot_test_data_overview, plot_raw_dataframe, plot_prediction_at_date, plot_multiple_predictions_at_date
from torch.utils.data import DataLoader
import torch
import argparse
import os


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
    print(f"📊 Loading data from {data_path}...")
    df = load_energy_hdf_to_pandas(data_path, plot_data=False)
    
    # Split (same splits as training)
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15)
    
    # Create datasets with scaler from training data
    print("⚙️  Creating datasets...")
    train_set = EnergyDataset(df_train, seq_len, output_horizon, normalize=True, scaler=None)
    test_set = EnergyDataset(df_test, seq_len, output_horizon, normalize=True, scaler=train_set.scaler)
    
    print(f"✅ Scaler info (fitted on training data):")
    print(f"   Min: {train_set.scaler.data_min_}")
    print(f"   Max: {train_set.scaler.data_max_}")
    print(f"   Train samples: {len(df_train)}")
    print(f"   Val samples: {len(df_val)}")
    print(f"   Test samples: {len(df_test)}")
    
    # Create DataLoader
    test_loader = DataLoader(test_set, batch_size=64)
    
    return test_loader, train_set.scaler, df_train, df_val, df_test


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
    print(f"🔄 Loading model from {model_path}...")
    
    # Initialize model architecture
    model = CNN_LSTM_Forecaster(
        input_dim=6, 
        seq_len=seq_len, 
        output_dim=output_horizon
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"✅ Model loaded successfully!")
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
    parser.add_argument('--no_overlap', action='store_true',
                        help='Use with --plot_full: only plot non-overlapping predictions (recommended!)')
    parser.add_argument('--skip_examples', action='store_true',
                        help='Skip individual example plots (useful with --plot_full)')
    parser.add_argument('--plot_data_only', action='store_true',
                        help='Plot only test data without predictions (no model loading needed)')
    parser.add_argument('--predict_at_date', type=str, default=None,
                        help='Make prediction starting at specific date (format: YYYY-MM-DD, e.g., 2018-08-31)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")
    
    # 1. Load test data
    test_loader, scaler, df_train, df_val, df_test = load_test_data(
        args.data_path, 
        args.seq_len, 
        args.output_horizon
    )
    
    # If only plotting data (no predictions), skip model loading
    if args.plot_data_only:
        print(f"\n📊 Plotting train/val/test datasets...")
        plot_raw_dataframe((df_train, df_val, df_test), title="Complete Dataset")
        print(f"\n✅ Done!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        print(f"   Please train a model first or specify correct path with --model_path")
        return
    
    # 2. Load model
    model = load_model(
        args.model_path,
        args.seq_len,
        args.output_horizon,
        device
    )
    
    # Special case: Prediction at specific date
    if args.predict_at_date:
        print(f"\n🎯 Making {args.n_examples} prediction(s) starting at: {args.predict_at_date}")
        if args.n_examples == 1:
            # Single prediction
            plot_prediction_at_date(
                model, 
                df_test, 
                args.predict_at_date, 
                seq_len=args.seq_len,
                output_horizon=args.output_horizon,
                scaler=scaler,
                device=device
            )
        else:
            # Multiple predictions spaced 12 hours apart
            plot_multiple_predictions_at_date(
                model,
                df_test,
                args.predict_at_date,
                n_examples=args.n_examples,
                seq_len=args.seq_len,
                output_horizon=args.output_horizon,
                scaler=scaler,
                device=device
            )
        print(f"\n✅ Done!")
        return
    
    # 3. Evaluate
    print(f"\n📈 Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n" + "="*50)
    print(f"TEST RESULTS")
    print(f"="*50)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.6f}")
    print(f"="*50)
    
    # 4. Plot predictions
    if args.plot_full:
        # Plot entire test set in one continuous plot
        if args.no_overlap:
            print(f"\n📊 Plotting entire test set (NON-OVERLAPPING)...")
            all_preds, all_targets = plot_full_test_set_non_overlapping(
                model, test_loader, device, 
                output_horizon=args.output_horizon,
                df_test=df_test,
                seq_len=args.seq_len,
                scaler=scaler
            )
        else:
            print(f"\n📊 Plotting entire test set (with overlaps)...")
            all_preds, all_targets = plot_full_test_set(model, test_loader, device)
        
    if not args.skip_examples:
        # Plot individual examples
        print(f"\n📊 Generating {args.n_examples} individual prediction examples...")
        plot_predictions(model, test_loader, device, n_examples=args.n_examples)
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
