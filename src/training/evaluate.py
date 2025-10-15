import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluates a trained model on the test set and computes metrics.

    Returns:
        metrics (dict): contains MSE, RMSE, MAE, MAPE
    """
    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss = criterion(y_pred, y_test)
            total_loss += loss.item() * X_test.size(0)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_test.cpu())

    total_loss /= len(test_loader.dataset)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(total_loss)
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

    print("\n" + "=" * 50)
    print(" TEST SET PERFORMANCE")
    print("=" * 50)
    print(f"MSE:   {total_loss:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE:   {mae:.6f}")
    print(f"MAPE:  {mape:.2f}%")
    print("=" * 50 + "\n")

    return {"MSE": total_loss, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot_predictions(model, test_loader, device="cpu", n_examples=3):
    """
    Plots example predictions from the test set.
    Shows input sequence, true output, and predicted output.
    """
    model.eval()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)

            for j in range(min(n_examples, X_test.size(0))):
                plt.figure(figsize=(12, 4))
                input_seq = X_test[j][:, -1].cpu().numpy()
                target_seq = y_test[j].cpu().numpy()
                pred_seq = y_pred[j].cpu().numpy()

                plt.plot(range(len(input_seq)), input_seq, label='Input sequence', color='blue')
                plt.plot(range(len(input_seq), len(input_seq) + len(target_seq)),
                         target_seq, 'r-o', label='True target')
                plt.plot(range(len(input_seq), len(input_seq) + len(pred_seq)),
                         pred_seq, 'g--x', label='Prediction')

                plt.xlabel("Timestep (15-min intervals)")
                plt.ylabel("Normalized Load")
                plt.title(f"Example Prediction {j+1}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            break