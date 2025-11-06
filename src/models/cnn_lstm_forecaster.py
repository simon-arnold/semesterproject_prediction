import torch
import torch.nn as nn


class CNN_LSTM_Forecaster(nn.Module):
    def __init__(self, input_dim=8, seq_len=192, output_dim=16):
        """
        CNN-LSTM model for short-term energy load forecasting.
        Args:
            input_dim (int): Number of input features per timestep.
                           Default 8 = Year (normalized)
                                     + 6 cyclic features (tod_sin/cos, weekday_sin/cos, doy_sin/cos)
                                     + Load (normalized, last feature)
            seq_len (int): Length of input sequence (timesteps).
            output_dim (int): Length of output vector (forecast horizon).
        """
        super().__init__()
        
        dropout_prob = 0.2

        # --- Convolutional feature extractor ---
        #Ich have das Gefühl kleinere kernel (die beiden ersten = 3) erkennen spitzen fast besser aber gewisse andere patterns werden schlechter erkannt
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=32,
            kernel_size=4, #Changed from 5->4 way faster training time
            stride=1,
            padding=2,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Previous configuration (64-100-128 filters without padding, each with pooling)
        # self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, stride=1)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=100, kernel_size=5, stride=1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # self.conv3 = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=5, stride=1)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Compute LSTM input size after Conv/Pool ---
        conv_out_len = self._calc_conv_output(seq_len)
        print(f"convolution out channels: {self.conv3.out_channels}, conv_out_len: {conv_out_len}")
        self.lstm_input_size = self.conv3.out_channels  # number of filters from last Conv layer

        # --- LSTM for temporal modeling ---
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=3*128, 
                           num_layers=1, batch_first=True, dropout=0.2)
        
        self.dropout_lstm = nn.Dropout(dropout_prob)

        # --- Fully connected layers ---
        # Use ALL lstm outputs, not just last timestep
        # self.fc1 = nn.Linear(128 * conv_out_len, 256)  # Flatten all LSTM outputs
        self.fc1 = nn.Linear(self.lstm.hidden_size, self.lstm.hidden_size)
        self.dropout_fc = nn.Dropout(dropout_prob)
        self.fc112 = nn.Linear(self.lstm.hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def _calc_conv_output(self, seq_len):
        """Helper to compute sequence length after three Conv+Pool stacks."""
        def conv1d_out(length, kernel_size, stride=1, padding=0, dilation=1):
            return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        def pool1d_out(length, kernel_size, stride=None, padding=0, dilation=1):
            stride = stride or kernel_size
            return ((length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        L = seq_len
        
        #current configuration
        L = conv1d_out(L, kernel_size=5, stride=1, padding=2)  # conv1
        L = conv1d_out(L, kernel_size=5, stride=1, padding=2)  # conv2
        L = pool1d_out(L, kernel_size=2, stride=2)
        L = conv1d_out(L, kernel_size=3, stride=1, padding=1)  # conv3
        L = pool1d_out(L, kernel_size=2, stride=2)
        
        #previous configuration
        # L = conv1d_out(L, kernel_size=5, stride=1, padding=0)  # conv1
        # L = pool1d_out(L, kernel_size=2, stride=2)             # pool1
        # L = conv1d_out(L, kernel_size=5, stride=1, padding=0)  # conv2
        # L = pool1d_out(L, kernel_size=2, stride=2)             # pool2
        # L = conv1d_out(L, kernel_size=5, stride=1, padding=0)  # conv3
        # L = pool1d_out(L, kernel_size=2, stride=2)             # pool3
        

        return L

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = x.permute(0, 2, 1)              # -> [batch, features, seq_len]

        # CNN feature extraction
        #current configuration
        x = (self.relu1(self.conv1(x)))
        x = (self.pool2(self.relu2(self.conv2(x))))
        x = (self.pool3(self.relu3(self.conv3(x))))

        # previous configuration
        # x = self.pool1(self.relu1(self.conv1(x)))
        # x = self.pool2(self.relu2(self.conv2(x)))
        # x = self.pool3(self.relu3(self.conv3(x)))

        # Prepare for LSTM: [batch, seq_len', features']
        x = x.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: [batch, seq_len', 128]
        
        # Use ALL timesteps, not just the last one!
        # This preserves temporal information throughout the sequence
        # x = lstm_out.reshape(lstm_out.size(0), -1)  # Flatten: [batch, seq_len' * 128]
        x = lstm_out[:, -1, :]  # Take only the last timestep output: [batch, 128]
        x = self.dropout_lstm(x)

        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = nn.functional.relu(self.fc112(x))
        x = nn.functional.relu(self.fc2(x))
        # out = self.fc3(x)  # Output layer
        out = torch.sigmoid(self.fc3(x))  # Sigmoid to constrain output to [0, 1]
        return out