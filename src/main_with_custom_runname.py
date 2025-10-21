# Beispiel: Wie man beschreibende Run-Namen verwendet

from data_processing.EnergyDataset import EnergyDataset
from data_processing.data_utils import load_energy_hdf_to_pandas, split_dataframe
from models.cnn_lstm_forecaster import CNN_LSTM_Forecaster
from training.train_loop import train_model
from training.evaluate import evaluate_model, plot_predictions
from torch.utils.data import DataLoader
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1️⃣ Load Data
    df = load_energy_hdf_to_pandas("data/data/dfA_300s.hdf", plot_data=False)

    # 2️⃣ Split
    df_train, df_val, df_test = split_dataframe(df, 0.7, 0.15, 0.15)

    # 3️⃣ Create datasets
    seq_len = 2*24*4
    output_horizon = 4*4
    train_set = EnergyDataset(df_train, seq_len, output_horizon)
    val_set = EnergyDataset(df_val, seq_len, output_horizon)
    test_set = EnergyDataset(df_test, seq_len, output_horizon)

    # 4️⃣ DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    # 5️⃣ Model + Training mit BESCHREIBENDEM RUN-NAMEN
    model = CNN_LSTM_Forecaster(input_dim=6, seq_len=seq_len, output_dim=output_horizon).to(device)
    
    # ========================================
    # OPTION 1: Beschreibender Name
    # ========================================
    run_name = "baseline_lr1e-4_bs64_ep10"
    log_dir = f"runs/{run_name}"
    
    model, best_val_loss = train_model(
        model, train_loader, val_loader, 
        n_epochs=10, 
        lr=1e-4, 
        device=device,
        log_dir=log_dir  # <-- Custom log directory
    )
    
    # Oder für Experimente:
    # run_name = "exp_higher_lr_1e-3"
    # run_name = "exp_larger_batch_128"
    # run_name = "exp_more_epochs_50"
    # run_name = "exp_2layer_lstm"
    
    # 6️⃣ Save model
    torch.save(model.state_dict(), "NN_weights/cnn_lstm_forecaster.pth")

    # 7️⃣ Evaluate
    test_metrics = evaluate_model(model, test_loader, device)
    plot_predictions(model, test_loader, device, n_examples=1)

if __name__ == "__main__":
    main()
