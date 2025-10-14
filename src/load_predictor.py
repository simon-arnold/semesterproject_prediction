# cnn_lstm_forecaster.py
import torch
import torch.nn as nn
import torch.optim as optim

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
    # Parameters
    batch_size = 8
    seq_len = 192       # 2 days with 15-min intervals
    n_features = 6      # Year, Month, Day, Timestep, Weekday, Load
    output_dim = 16     # 4x4 flattened

    # Dummy input data (random)
    X = torch.randn(batch_size, seq_len, n_features)
    y_true = torch.randn(batch_size, output_dim)

    # Initialize model, loss, optimizer
    model = CNN_LSTM_Forecaster(input_dim=n_features, seq_len=seq_len, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y_true)

    # Backward + Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print output
    print(f"Input shape : {X.shape}")
    print(f"Output shape: {y_pred.shape}")
    print(f"Training loss: {loss.item():.6f}")
