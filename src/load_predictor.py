# cnn_lstm_forecaster.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_processing.EnergyDataset import EnergyDataset, load_energy_hdf_to_pandas


from rich.traceback import install
install(show_locals=False)


# -------------------------------------------------------------
# CNN-LSTM Predictor Definition
# -------------------------------------------------------------



# -------------------------------------------------------------
# Run Example
# -------------------------------------------------------------
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Load and preprocess data ---
    file_path = "data/data/dfA_300s.hdf"
    df = load_energy_hdf_to_pandas(file_path, plot_data=False)

    # --- 2. Chronological Train / Val / Test split ---
    n = len(df)
    
    train_frac = 0.7
    val_frac   = 0.15
    test_frac  = 0.15

    ntrain = int(n * train_frac)
    nval   = int(n * val_frac)
    ntest  = n - ntrain - nval

    df_train = df.iloc[:ntrain]
    df_val   = df.iloc[ntrain:ntrain+nval]
    df_test  = df.iloc[ntrain+nval:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(df_train)} samples ({train_frac*100:.0f}%)")
    print(f"  Val:   {len(df_val)} samples ({val_frac*100:.0f}%)")
    print(f"  Test:  {len(df_test)} samples ({test_frac*100:.0f}%)")

    # --- 3. Dataset objects ---
    seq_len = 2*24*4  # 2 days of 15-min timesteps 
    output_horizon = 4*4  # next 4 hours (16 steps)

    train_dataset = EnergyDataset(df_train, seq_len=seq_len, output_horizon=output_horizon)
    val_dataset   = EnergyDataset(df_val, seq_len=seq_len, output_horizon=output_horizon)
    test_dataset  = EnergyDataset(df_test, seq_len=seq_len, output_horizon=output_horizon)
    
    print(f"\nSequence generation:")
    print(f"  Input sequence length: {seq_len} timesteps ({seq_len/4:.1f} hours)")
    print(f"  Output horizon: {output_horizon} timesteps ({output_horizon/4:.1f} hours)")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")

    # --- 4. DataLoaders ---
    batch_size = 8*2
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(" -------DataLoader Constructred------- ")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # --- 5. Model, Loss, Optimizer ---
    n_features = 6
    output_dim = output_horizon

    model = CNN_LSTM_Forecaster(input_dim=n_features, seq_len=seq_len, output_dim=output_dim)
    model = model.to(device)
    criterion = torch.nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 6. Training loop ---
    n_epochs = 10  
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                y_pred_val = model(X_val)
                loss_val = criterion(y_pred_val, y_val)
                val_loss += loss_val.item() * X_val.size(0)
        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
    # --- 7. Save model ---
    save_path = "NN_weights/cnn_lstm_forecaster.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_len': seq_len,
        'output_dim': output_dim,
        'n_features': n_features
    }, save_path)
    print(f"Model saved to {save_path}")
    
    
    # --- 8. TEST SECTION ---
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_pred = model(X_test)
            loss = criterion(y_pred, y_test)
            test_loss += loss.item() * X_test.size(0)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_test.cpu())

    test_loss /= len(test_dataset)
    
    # Calculate additional metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    mae = np.mean(np.abs(all_preds - all_targets))
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100
    rmse = np.sqrt(test_loss)
    
    print(f"\n{'='*50}")
    print("TEST SET PERFORMANCE:")
    print(f"{'='*50}")
    print(f"MSE:  {test_loss:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*50}\n")

    n_examples = 1  # wie viele Testbeispiele du sehen willst

    model.eval()
    with torch.no_grad():
        for i, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test)
            
            for j in range(min(n_examples, X_test.size(0))):
                plt.figure(figsize=(12, 4))
                
                # Eingabesequenz (letzter Kanal = Load)
                input_seq = X_test[j][:, -1].cpu().numpy()
                target_seq = y_test[j].cpu().numpy()
                pred_seq = y_pred[j].cpu().numpy()

                plt.plot(range(len(input_seq)), input_seq, label='Input Load sequence', color='blue')
                plt.plot(range(len(input_seq), len(input_seq) + len(target_seq)),
                        target_seq, 'r-o', label='True Target horizon')
                plt.plot(range(len(input_seq), len(input_seq) + len(pred_seq)),
                        pred_seq, 'g--x', label='Predicted horizon')

                plt.title(f"Test Sample {j} (Batch {i})")
                plt.xlabel("Timestep (15-min intervals)")
                plt.ylabel("Normalized Load")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            break
            
    
    print(f"Batch input shape : {X_batch.shape}")
    print(f"Batch output shape: {y_pred.shape}")
    print(f"Training loss: {loss.item():.6f}")
