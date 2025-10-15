# cnn_lstm_forecaster.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_processing.EnergyDataset import EnergyDataset, load_energy_hdf_to_pandas


from rich.traceback import install
install(show_locals=True)


# -------------------------------------------------------------
# CNN-LSTM Predictor Definition
# -------------------------------------------------------------
class CNN_LSTM_Forecaster(nn.Module):
    def __init__(self, input_dim=6, seq_len=192, output_dim=16):
        """
        CNN-LSTM model for short-term energy load forecasting.
        Args:
            input_dim (int): Number of input features per timestep.
            seq_len (int): Length of input sequence (timesteps).
            output_dim (int): Length of output vector (forecast horizon).
        """
        super().__init__()

        # --- Convolutional feature extractor ---
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Compute LSTM input size after Conv/Pool ---
        conv_out_len = self._calc_conv_output(seq_len)
        self.lstm_input_size = 64  # number of filters from last Conv layer

        # --- LSTM for temporal modeling ---
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=64, batch_first=True)

        # --- Fully connected layers ---
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.tanh = nn.Tanh()

    def _calc_conv_output(self, seq_len):
        """Helper to compute sequence length after two Conv+Pool stacks."""
        L = seq_len
        L = (L - 2 + 1)  # conv1
        L = L // 2       # pool1
        L = (L - 2 + 1)  # conv2
        L = L // 2       # pool2
        return L

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = x.permute(0, 2, 1)              # -> [batch, features, seq_len]

        # CNN feature extraction
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Prepare for LSTM: [batch, seq_len', features']
        x = x.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        x = self.tanh(lstm_out[:, -1, :])   # take last timestep

        # Fully connected layers
        x = self.fc1(x)
        out = self.fc2(x)
        return out


# -------------------------------------------------------------
# Run Example
# -------------------------------------------------------------
if __name__ == "__main__":
    
    
    # --- 1. Load and preprocess data ---
    file_path = "data/data/dfA_300s.hdf"
    df = load_energy_hdf_to_pandas(file_path, plot_data=False)

    # --- 2. Chronological Train / Val / Test split ---
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    df_train = df.iloc[:train_end]
    df_val   = df.iloc[train_end:val_end]
    df_test  = df.iloc[val_end:]

    # --- 3. Dataset objects ---
    seq_len = 192
    output_horizon = 16

    train_dataset = EnergyDataset(df_train, seq_len=seq_len, output_horizon=output_horizon)
    val_dataset   = EnergyDataset(df_val, seq_len=seq_len, output_horizon=output_horizon)
    test_dataset  = EnergyDataset(df_test, seq_len=seq_len, output_horizon=output_horizon)
    
    #train_dataset.print_info(n_examples=1)

    # --- 4. DataLoaders ---
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(" -------DataLoader Constructred------- ")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # --- 5. Model, Loss, Optimizer ---
    n_features = 6
    output_dim = output_horizon

    model = CNN_LSTM_Forecaster(input_dim=n_features, seq_len=seq_len, output_dim=output_dim)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 6. Single Training Step Example ---
    X_batch, y_batch = next(iter(train_loader))
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Batch input shape : {X_batch.shape}")
    print(f"Batch output shape: {y_pred.shape}")
    print(f"Training loss: {loss.item():.6f}")
