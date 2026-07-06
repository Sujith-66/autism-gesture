"""
train.py
---------
Paper Reference: Lakkapragada et al., JMIR Biomed Eng 2022;7(1):e33771

Trains an LSTM classifier on MobileNetV2 features extracted from SSBD.

Architecture (exactly as in paper):
    Input  : (batch, 90, 1280)   ← 90-frame sequences, 1280-dim MobileNetV2 features
    LSTM   : hidden_size=64, batch_first=True
    Dropout: p=0.30
    Dense  : output_size=3  (arm_flapping | head_banging | spinning)
    Loss   : CrossEntropyLoss
    Optim  : Adam, lr=0.01
    Epochs : 75  (paper: "trained until convergence for 10+ epochs → 75 total")

Evaluation:
    5-fold CV × N_SEEDS random seeds → N_SEEDS*5 folds total
    Reports mean ± SD of accuracy, precision, recall, F1

Usage:
    python train.py --X X.npy --y y.npy --seeds 10 --epochs 75
    (Use --seeds 100 to replicate the paper exactly — takes longer)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model ────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Exactly the architecture described in the paper:
      LSTM(64) → Dropout(0.30) → Linear(num_classes)

    Paper quote: "LSTM layer with a 64-dimensional output … inserted a dropout
    layer … with a dropout rate of 30%."
    """
    def __init__(self, input_size: int = 1280,
                       hidden_size: int = 64,
                       num_classes: int = 3,
                       dropout: float = 0.30):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _  = self.lstm(x)          # (B, T, 64)
        out     = out[:, -1, :]         # last timestep → (B, 64)
        out     = self.dropout(out)
        return self.fc(out)             # (B, num_classes)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_val, y_val,
                   device, epochs=75, lr=0.01, batch_size=16):
    """
    Train one fold; returns best-epoch predictions on the validation split.
    Paper: "reverted the model's weights to its weights for which it performed best."
    """
    model     = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tr_ds  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(y_tr, dtype=torch.long))
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_f1    = -1.0
    best_val_preds = None
    best_state     = None

    for epoch in range(epochs):
        model.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        # -- Validation --
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()

        fold_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if fold_f1 > best_val_f1:
            best_val_f1    = fold_f1
            best_val_preds = preds.copy()
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}

    return best_val_preds, best_state


# ── K-Fold CV over multiple seeds ─────────────────────────────────────────────
def cross_validate(X, y, n_splits=5, n_seeds=10, epochs=75, device=None):
    """
    Replicates the paper's evaluation:
        5-fold CV × n_seeds random seeds → n_seeds * n_splits folds total.
    Reports mean ± SD of accuracy, precision, recall, F1 (macro).
    """
    if device is None:
        device = torch.device("cpu")

    metrics = {"acc": [], "prec": [], "rec": [], "f1": []}

    print(f"\n[CV] {n_splits}-fold × {n_seeds} seeds = {n_splits * n_seeds} total folds\n")

    for seed in range(n_seeds):
        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr   = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            preds, _ = train_one_fold(X_tr, y_tr, X_val, y_val,
                                      device=device, epochs=epochs)

            acc  = accuracy_score(y_val, preds) * 100
            prec = precision_score(y_val, preds, average="macro", zero_division=0) * 100
            rec  = recall_score(y_val, preds, average="macro", zero_division=0) * 100
            f1   = f1_score(y_val, preds, average="macro", zero_division=0) * 100

            metrics["acc"].append(acc)
            metrics["prec"].append(prec)
            metrics["rec"].append(rec)
            metrics["f1"].append(f1)

            print(f"  seed={seed:02d} fold={fold+1}/{n_splits} | "
                  f"acc={acc:.1f}  prec={prec:.1f}  rec={rec:.1f}  F1={f1:.1f}")

    print("\n" + "="*62)
    print("FINAL RESULTS  (mean ± SD across all folds)")
    print("="*62)
    for key, vals in metrics.items():
        arr = np.array(vals)
        print(f"  {key.upper():5s} : {arr.mean():.1f} ± {arr.std():.1f}")
    print("="*62)

    return metrics


# ── Train final model on full data ────────────────────────────────────────────
def train_final_model(X, y, device, epochs=75, save_path="lstm_model.pth"):
    """
    Train on the entire dataset (no hold-out) and save.
    Used for the inference server (app.py).
    """
    print("\n[TRAIN] Training final model on full dataset …")
    set_seed(42)

    model     = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    best_loss  = float("inf")
    best_state = None

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dl)
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 15 == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"[DONE] Model saved → {save_path}  (best loss={best_loss:.4f})")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train LSTM on SSBD features")
    parser.add_argument("--X",        type=str, default="X.npy")
    parser.add_argument("--y",        type=str, default="y.npy")
    parser.add_argument("--seeds",    type=int, default=10,
                        help="Number of CV seeds (paper uses 100)")
    parser.add_argument("--epochs",   type=int, default=75)
    parser.add_argument("--save",     type=str, default="lstm_model.pth")
    parser.add_argument("--skip_cv",  action="store_true",
                        help="Skip cross-validation; just train final model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")

    X = np.load(args.X)
    y = np.load(args.y)
    print(f"[INFO] X shape: {X.shape}   y shape: {y.shape}")
    print(f"[INFO] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    if not args.skip_cv:
        cross_validate(X, y, n_splits=5, n_seeds=args.seeds,
                       epochs=args.epochs, device=device)

    train_final_model(X, y, device=device, epochs=args.epochs, save_path=args.save)


if __name__ == "__main__":
    main()
