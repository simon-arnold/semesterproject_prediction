# src/training/train_loop.py

import torch
from torch import nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-4, device="cpu", save_path="NN_weights/best_model.pth"):
    """
    Trains a PyTorch model using given DataLoaders.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: number of epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
        save_path: path to save the best model

    Returns:
        model (with best weights)
        best_val_loss (float)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # Mean Absolute Error

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                loss_val = criterion(y_pred, y_val)
                val_loss += loss_val.item() * X_val.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved ({save_path})")

    print(f"\nTraining finished. Best Validation Loss: {best_val_loss:.6f}")

    # Load best weights back
    model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, best_val_loss
