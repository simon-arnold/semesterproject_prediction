# src/training/train_loop.py

import torch
from torch import nn
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


def train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-4, device="cpu", 
                save_path="NN_storage/NN_weights/best_model.pth", log_dir=None, use_scheduler=True):
    """
    Trains a PyTorch model using given DataLoaders with TensorBoard logging.
    Optionally includes ReduceLROnPlateau scheduler for automatic learning rate adjustment.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        n_epochs: number of epochs
        lr: initial learning rate (will be reduced automatically if use_scheduler=True)
        device: 'cuda' or 'cpu'
        save_path: path to save the best model
        log_dir: TensorBoard log directory (None = auto-generate with timestamp)
        use_scheduler: if True, uses ReduceLROnPlateau to adjust LR (default: True)

    Returns:
        model (with best weights)
        best_val_loss (float)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error - better for regression
    
    # Learning Rate Scheduler (optional): Reduces LR when validation loss plateaus
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',           # Minimize validation loss
            factor=0.5,           # Reduce LR to 50% of current value
            patience=5,           # Wait 5 epochs without improvement
            verbose=True,         # Print message when LR is reduced
            min_lr=1e-7          # Don't reduce LR below this value
        )
        print("📉 LR Scheduler enabled (ReduceLROnPlateau: factor=0.5, patience=5)")
    else:
        print("📊 LR Scheduler disabled (constant learning rate)")

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Create TensorBoard writer with timestamp
    if log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"runs/load_predictor_{timestamp}"
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"   Run: tensorboard --logdir=runs")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Log model architecture
    try:
        # Get input_dim dynamically from the model's first conv layer
        # Works for both cyclic (8 features) and raw (6 features) modes
        input_dim = model.conv1.in_channels
        dummy_input = torch.randn(1, train_loader.dataset.seq_len, input_dim).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"⚠️  Could not log model graph: {e}")
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', 
                    f'lr={lr}, n_epochs={n_epochs}, batch_size={train_loader.batch_size}')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_text('Model', f'Total params: {total_params:,}, Trainable: {trainable_params:,}')

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
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalars('Loss/train_vs_val', {
            'train': train_loss,
            'validation': val_loss
        }, epoch)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Log gradient norms (for debugging)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar('Gradient/norm', total_norm, epoch)

        print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        # Update learning rate based on validation loss (if scheduler is enabled)
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            writer.add_scalar('Best_Val_Loss', best_val_loss, epoch)
            print(f"  ✅ New best model saved ({save_path})")

    final_lr = optimizer.param_groups[0]['lr']
    print(f"\nTraining finished. Best Validation Loss: {best_val_loss:.6f}")
    print(f"Final Learning Rate: {final_lr:.2e} (started at {lr:.2e})")
    
    # Log final hyperparameters with results
    writer.add_hparams(
        {
            'lr': lr,
            'batch_size': train_loader.batch_size,
            'n_epochs': n_epochs
        },
        {
            'hparam/best_val_loss': best_val_loss,
            'hparam/final_train_loss': train_losses[-1],
        }
    )
    
    # Close the writer
    writer.close()
    print(f"📊 TensorBoard logs saved. View with: tensorboard --logdir=runs")

    # Load best weights back
    model.load_state_dict(torch.load(save_path, weights_only=True, map_location=device))

    return model, best_val_loss
