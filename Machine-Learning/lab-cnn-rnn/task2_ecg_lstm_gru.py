import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ecg5000_series(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)

    train_features = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    test_features = test_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    train_series = train_features.reshape(-1)
    test_series = test_features.reshape(-1)

    mean = train_series.mean()
    std = train_series.std() + 1e-8
    train_series = (train_series - mean) / std
    test_series = (test_series - mean) / std

    return train_series, test_series, mean, std


def make_windows(series: np.ndarray, input_len: int, pred_len: int, stride: int):
    x, y = [], []
    end = len(series) - input_len - pred_len + 1
    for i in range(0, end, stride):
        x.append(series[i : i + input_len])
        y.append(series[i + input_len : i + input_len + pred_len])
    x = np.asarray(x, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.float32)
    return x, y


class RNNForecaster(nn.Module):
    def __init__(self, model_type: str, hidden_size: int, num_layers: int, dropout: float, pred_len: int):
        super().__init__()
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.GRU
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def choose_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total += y_batch.size(0)
    return total_loss / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    preds = []
    trues = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            total += y_batch.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae = np.mean(np.abs(preds - trues))
    return float(total_loss / total), float(mae), trues, preds


def save_plots(trues, preds, mean, std, run_dir: Path, plot_points: int):
    trues_denorm = trues * std + mean
    preds_denorm = preds * std + mean

    first_step_true = trues_denorm[:, 0]
    first_step_pred = preds_denorm[:, 0]
    n = min(plot_points, len(first_step_true))

    plt.figure(figsize=(12, 4))
    plt.plot(first_step_true[:n], label="Actual (t+1)")
    plt.plot(first_step_pred[:n], label="Predicted (t+1)")
    plt.title("First-Step Forecast Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "first_step_curve.png")
    plt.close()

    sample_idx = min(len(trues_denorm) - 1, 20)
    horizon_true = trues_denorm[sample_idx]
    horizon_pred = preds_denorm[sample_idx]
    horizon = np.arange(1, len(horizon_true) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(horizon, horizon_true, marker="o", label="Actual horizon")
    plt.plot(horizon, horizon_pred, marker="x", label="Predicted horizon")
    plt.title("Multi-Step Horizon Forecast Sample")
    plt.xlabel("Forecast step")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "horizon_sample.png")
    plt.close()


def run_single_model(model_name, args, train_loader, test_loader, mean, std, device, output_root: Path):
    run_dir = output_root / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RNNForecaster(
        model_type=model_name,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pred_len=args.pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = choose_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    best_mse = float("inf")
    history = []
    best_path = run_dir / "best_model.pth"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_mse, test_mae, _, _ = evaluate(model, test_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_mse": float(train_loss),
            "test_mse": float(test_mse),
            "test_mae": float(test_mae),
        })

        if test_mse < best_mse:
            best_mse = test_mse
            torch.save(model.state_dict(), best_path)

        print(
            f"[{model_name}] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train MSE {train_loss:.6f} | Test MSE {test_mse:.6f} | Test MAE {test_mae:.6f}"
        )

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_mse, final_mae, trues, preds = evaluate(model, test_loader, criterion, device)
    save_plots(trues, preds, mean, std, run_dir, args.plot_points)

    metrics = {
        "model": model_name,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "best_test_mse": float(best_mse),
        "final_test_mse": float(final_mse),
        "final_test_mae": float(final_mae),
        "train_seconds": float(time.time() - start),
        "history": history,
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: ECG5000 multi-step forecasting with LSTM/GRU")
    parser.add_argument("--model", choices=["lstm", "gru", "both"], default="both")
    parser.add_argument("--train_path", type=str, default="./ECG5000/ECG5000_TRAIN.tsv")
    parser.add_argument("--test_path", type=str, default="./ECG5000/ECG5000_TEST.tsv")
    parser.add_argument("--input_len", type=int, default=60, help="Input window size m")
    parser.add_argument("--pred_len", type=int, default=10, help="Prediction horizon p (1=single-step)")
    parser.add_argument("--stride", type=int, default=5, help="Window sampling stride")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--plot_points", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="./outputs/task2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_series, test_series, mean, std = load_ecg5000_series(Path(args.train_path), Path(args.test_path))
    x_train, y_train = make_windows(train_series, args.input_len, args.pred_len, args.stride)
    x_test, y_test = make_windows(test_series, args.input_len, args.pred_len, args.stride)

    train_dataset = SequenceDataset(x_train, y_train)
    test_dataset = SequenceDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    models_to_run = [args.model] if args.model != "both" else ["lstm", "gru"]
    summaries = []
    for model_name in models_to_run:
        summaries.append(run_single_model(model_name, args, train_loader, test_loader, mean, std, device, output_root))

    if len(summaries) == 2:
        comparison = {
            "task": "ecg5000_lstm_gru_multistep_comparison",
            "lstm": summaries[0],
            "gru": summaries[1],
            "better_model": summaries[0]["model"] if summaries[0]["final_test_mse"] <= summaries[1]["final_test_mse"] else summaries[1]["model"],
        }
        with (output_root / "comparison.json").open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison written to {output_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
