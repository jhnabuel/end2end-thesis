import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_report(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return

    with open(json_path, 'r') as f:
        report = json.load(f)

    history = report["epoch_history"]
    epochs = [row["epoch"] for row in history]
    
    # Extract metrics
    train_mse = [row["train_mse"] for row in history]
    val_mse = [row["val_mse"] for row in history]
    train_mae = [row["train_mae"] for row in history]
    val_mae = [row["val_mae"] for row in history]
    
    best_epoch = report["training_results"]["best_epoch"]

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"Training Dashboard: {report['run_metadata']['timestamp']}", fontsize=16)

    # --- Panel 1: MSE Loss (The Optimization Metric) ---
    axes[0].plot(epochs, train_mse, label='Train MSE', color='blue', linewidth=2)
    axes[0].plot(epochs, val_mse, label='Validation MSE', color='orange', linewidth=2)
    axes[0].axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[0].set_title('Mean Squared Error (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: MAE (The Human-Readable Metric) ---
    axes[1].plot(epochs, train_mae, label='Train MAE', color='green', linewidth=2)
    axes[1].plot(epochs, val_mae, label='Validation MAE', color='purple', linewidth=2)
    axes[1].axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
    axes[1].set_title('Mean Absolute Error (Steering Accuracy)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (Normalized Steering Range)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save the plot next to the json file
    plot_out = json_path.replace('.json', '.png')
    plt.savefig(plot_out, dpi=300)
    print(f"Plot saved successfully to: {plot_out}")
    
    # Show it on the screen
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DAVE-2 Training Metrics")
    parser.add_argument("json_file", help="Path to the training_report.json file")
    args = parser.parse_args()
    
    plot_training_report(args.json_file)