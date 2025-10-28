from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe
from models.cnn_lstm_forecaster import CNN_LSTM_Forecaster
from training.train_loop import train_model
from training.evaluate import evaluate_model, plot_multiple_predictions_at_date
from torch.utils.data import DataLoader
import torch
import os

YELLOW = '\033[93m'
RESET = '\033[0m'

store_onnx = True  

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1️. Load Data
    df = load_energy_hdf_to_pandas("data/data/dfA_300s.hdf", plot_data=False)

    # 2️. Split
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15)

    # 3️. Create datasets
    seq_len = 2*24*4
    output_horizon = 4*4
    
    # IMPORTANT: Fit scaler on training data, then reuse for val/test!
    train_set = EnergyDataset(df_train, seq_len, output_horizon, normalize=True, scaler=None)
    val_set = EnergyDataset(df_val, seq_len, output_horizon, normalize=True, scaler=train_set.scaler)
    test_set = EnergyDataset(df_test, seq_len, output_horizon, normalize=True, scaler=train_set.scaler)
    
    print(f"\n✅ Scaler fitted on training data:")
    print(f"   Min values: {train_set.scaler.data_min_}")
    print(f"   Max values: {train_set.scaler.data_max_}")

    # 4️. DataLoaders
    batch_size = 32
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    # 5️. Model + Training
    training_epochs = 1*10
    # input_dim = 8 features:
    #   - Year (normalized)
    #   - tod_sin, tod_cos (time of day, cyclic)
    #   - weekday_sin, weekday_cos (day of week, cyclic)
    #   - doy_sin, doy_cos (day of year, cyclic)
    #   - Load (normalized, last feature)
    model = CNN_LSTM_Forecaster(input_dim=8, seq_len=seq_len, output_dim=output_horizon).to(device)
    model, best_val_loss = train_model(model, train_loader, val_loader, n_epochs=training_epochs, lr=1e-4, device=device)

    # 6️. Save model (PyTorch format)
    save_dir = "NN_storage/NN_weights"
    os.makedirs(save_dir, exist_ok=True)
    
    pth_path = os.path.join(save_dir, "cnn_lstm_forecaster.pth")
    torch.save(model.state_dict(), pth_path)
    print(f" Model saved as PyTorch weights: {pth_path}")
    
    
    if store_onnx:
        # 6b. Save model as ONNX (directly in NN_Storage/)
        onnx_dir = "NN_storage"
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, "cnn_lstm_forecaster.onnx")
        
        # Set model to eval mode for export
        model.eval()
        
        # Create dummy input with correct shape [batch_size, seq_len, num_features]
        # num_features = 8 (Year + 6 cyclic + Load)
        dummy_input = torch.randn(1, seq_len, 8).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,                          # model being run
            dummy_input,                    # model input (or a tuple for multiple inputs)
            onnx_path,                      # where to save the model
            export_params=True,             # store the trained parameter weights
            opset_version=14,               # ONNX version to export to
            do_constant_folding=True,       # optimize constant folding
            input_names=['input'],          # model's input names
            output_names=['output'],        # model's output names
            dynamic_axes={                  # variable length axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✅ Model exported as ONNX: {onnx_path}")
        print(f"   Input shape: [batch_size, {seq_len}, 8]  (8 features: Year + 6 cyclic + Load)")
        print(f"   Output shape: [batch_size, {output_horizon}]")

    # 7️. Evaluate
    test_metrics = evaluate_model(model, test_loader, device)
    plot_multiple_predictions_at_date(model=model, test_set=test_set, device=device)
    
        
    # Wait for user input before closing
    input(f"\n{YELLOW}Press Enter to exit...{RESET}")

if __name__ == "__main__":
    main()
