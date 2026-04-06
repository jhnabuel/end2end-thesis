"""
trainer.py  –  DAVE2 end-to-end training for autonomous robot steering.

Logs captured during training
──────────────────────────────
  Per-epoch (printed + CSV):
    • epoch number / total
    • train MSE loss (normalised steering, range ≈ 0-2)
    • val   MSE loss
    • train MAE  (mean absolute steering error in normalised units)
    • val   MAE
    • epoch wall-clock time (seconds)
    • learning rate (supports schedulers)

  After training (JSON summary):
    • run metadata  – timestamp, hardware, Python/PyTorch versions
    • dataset info  – catalog path, total samples, train/val split, batch size
    • model info    – architecture name, total parameters, trainable parameters
    • hyperparameters – epochs, lr, optimiser, loss function, scheduler
    • per-epoch history  (list of dicts, same columns as CSV)
    • best epoch & metrics
    • training duration (wall-clock, hh:mm:ss)
    • checkpoint paths

Output files (written next to trainer.py)
──────────────────────────────────────────
  dave2_robot_model.pth                   – best-val-loss model weights (weights only)
  checkpoints_<timestamp>/
    ckpt_epoch_<N>.pt                     – full checkpoint every `checkpoint_every` epochs
                                            (model + optimizer + scheduler + metadata)
    ckpt_best.pt                          – always the best-val checkpoint (full state)
  training_log_<timestamp>.csv            – epoch-by-epoch table
  training_report_<timestamp>.json        – full thesis summary
"""

import json
import os
import sys
import time
import platform
import csv
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Path fix – importable from any working directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model import DAVE2
from dataset import SelfDrivingDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def seconds_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def compute_mae(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean Absolute Error on a batch (both in normalised steering space)."""
    return torch.mean(torch.abs(outputs - labels)).item()


def evaluate(model, dataloader, criterion, device):
    """Run one pass over *dataloader* and return (avg_mse, avg_mae)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            total_mae += compute_mae(outputs, labels)
    n = len(dataloader)
    return total_loss / n, total_mae / n


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    catalog_file: str = "../data/catalog_0.catalog",
    image_dir: str = "../data/",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    val_split: float = 0.15,
    num_workers: int = 4,
    output_dir: str = _THIS_DIR,
    checkpoint_every: int = 5,   # save a full checkpoint every N epochs (0 = disable)
):
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    weights_out  = os.path.join(output_dir, "dave2_robot_model.pth")
    csv_out      = os.path.join(output_dir, f"training_log_{run_ts}.csv")
    json_out     = os.path.join(output_dir, f"training_report_{run_ts}.json")
    ckpt_dir     = os.path.join(output_dir, f"checkpoints_{run_ts}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset & split
    # ------------------------------------------------------------------
    full_dataset = SelfDrivingDataset(catalog_file, image_dir)
    total_samples = len(full_dataset)

    val_size   = max(1, int(total_samples * val_split))
    train_size = total_samples - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"[DATA]  Total: {total_samples}  |  Train: {train_size}  |  Val: {val_size}")

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")

    model = DAVE2().to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[MODEL]  Total params: {total_params:,}  |  Trainable: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ReduceLROnPlateau: halve LR if val loss stagnates for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ------------------------------------------------------------------
    # CSV initialisation
    # ------------------------------------------------------------------
    csv_columns = [
        "epoch", "train_mse", "val_mse",
        "train_mae", "val_mae",
        "lr", "epoch_time_s",
    ]
    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    best_val_mse = float("inf")
    best_epoch   = 0
    training_start = time.time()

    print(f"\n{'─'*65}")
    print(f"  Epoch  │  Train MSE  │  Val MSE  │  Train MAE │  Val MAE  │  LR")
    print(f"{'─'*65}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_mae  = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_mae  += compute_mae(outputs.detach(), labels)

        train_mse = running_loss / len(train_loader)
        train_mae = running_mae  / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────
        val_mse, val_mae = evaluate(model, val_loader, criterion, device)

        # ── Scheduler step ─────────────────────────────────────────────
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # ── Save best model (weights-only .pth + full best checkpoint) ──
        is_best = val_mse < best_val_mse
        if is_best:
            best_val_mse = val_mse
            best_epoch   = epoch
            torch.save(model.state_dict(), weights_out)
            torch.save({
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "val_mse":          val_mse,
                "val_mae":          val_mae,
                "train_mse":        train_mse,
                "train_mae":        train_mae,
                "lr":               current_lr,
                "run_ts":           run_ts,
            }, os.path.join(ckpt_dir, "ckpt_best.pt"))

        # ── Periodic checkpoint every N epochs ────────────────────────
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "scheduler_state":  scheduler.state_dict(),
                "val_mse":          val_mse,
                "val_mae":          val_mae,
                "train_mse":        train_mse,
                "train_mae":        train_mae,
                "lr":               current_lr,
                "run_ts":           run_ts,
            }, ckpt_path)
            print(f"  [CKPT] Saved periodic checkpoint → {ckpt_path}")

        # ── Log ────────────────────────────────────────────────────────
        row = {
            "epoch":        epoch,
            "train_mse":    round(train_mse, 6),
            "val_mse":      round(val_mse,   6),
            "train_mae":    round(train_mae,  6),
            "val_mae":      round(val_mae,    6),
            "lr":           current_lr,
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        best_marker = " ◀ best" if epoch == best_epoch else ""
        print(
            f"  {epoch:>5}  │  {train_mse:.5f}    │  {val_mse:.5f}  │"
            f"  {train_mae:.5f}  │  {val_mae:.5f}  │  {current_lr:.2e}"
            f"  {best_marker}"
        )

    # ------------------------------------------------------------------
    # Close CSV
    # ------------------------------------------------------------------
    csv_file.close()
    total_time = time.time() - training_start

    print(f"{'─'*65}")
    print(f"\n[DONE]  Best val MSE: {best_val_mse:.6f} at epoch {best_epoch}")
    print(f"[DONE]  Total training time: {seconds_to_hms(total_time)}")
    print(f"[DONE]  Model saved  → {weights_out}")
    print(f"[DONE]  CSV log      → {csv_out}")

    # ------------------------------------------------------------------
    # JSON summary report
    # ------------------------------------------------------------------
    best_row = history[best_epoch - 1]
    report = {
        "run_metadata": {
            "timestamp":        run_ts,
            "python_version":   platform.python_version(),
            "pytorch_version":  torch.__version__,
            "cuda_available":   torch.cuda.is_available(),
            "cuda_device":      torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform":         platform.platform(),
            "processor":        platform.processor(),
        },
        "dataset": {
            "catalog_file":     catalog_file,
            "image_dir":        image_dir,
            "total_samples":    total_samples,
            "train_samples":    train_size,
            "val_samples":      val_size,
            "val_split_ratio":  val_split,
            "batch_size":       batch_size,
            "num_workers":      num_workers,
            "input_shape":      [3, 66, 200],
            "label":            "normalised steering angle  (raw_angle / 50.0)",
            "label_range":      [-1.0, 1.0],
        },
        "model": {
            "architecture":         "DAVE2 (NVIDIA end-to-end CNN)",
            "total_parameters":     total_params,
            "trainable_parameters": trainable_params,
            "output_activation":    "tanh",
            "internal_normalisation": "(x / 127.5) - 1.0",
        },
        "hyperparameters": {
            "epochs":             epochs,
            "learning_rate":      lr,
            "optimiser":          "Adam",
            "loss_function":      "MSELoss",
            "scheduler":          "ReduceLROnPlateau (factor=0.5, patience=3)",
            "dropout":            0.5,
            "checkpoint_every_n_epochs": checkpoint_every,
        },
        "training_results": {
            "best_epoch":            best_epoch,
            "best_val_mse":          round(best_val_mse, 6),
            "best_val_mae":          round(best_row["val_mae"], 6),
            "best_train_mse":        round(best_row["train_mse"], 6),
            "best_train_mae":        round(best_row["train_mae"], 6),
            "final_lr":              optimizer.param_groups[0]['lr'],
            "total_epochs_run":      epochs,
            "training_duration_s":   round(total_time, 2),
            "training_duration_hms": seconds_to_hms(total_time),
        },
        "output_files": {
            "model_weights":     weights_out,
            "best_checkpoint":   os.path.join(ckpt_dir, "ckpt_best.pt"),
            "checkpoint_dir":    ckpt_dir,
            "epoch_csv_log":     csv_out,
            "json_report":       json_out,
        },
        "epoch_history": history,
    }

    with open(json_out, "w") as jf:
        json.dump(report, jf, indent=2)

    print(f"[DONE]  JSON report  → {json_out}")
    return report


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()