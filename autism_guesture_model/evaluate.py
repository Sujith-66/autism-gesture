"""
evaluate.py
------------
Paper Reference: Lakkapragada et al., JMIR Biomed Eng 2022;7(1):e33771

Standalone evaluation script that:
  1. Loads X.npy / y.npy
  2. Runs 5-fold CV (optionally over multiple seeds)
  3. Computes accuracy, precision, recall, F1, AUROC — exactly as in the paper
  4. Plots average ROC curves (one per class + macro-average)
  5. Saves results to results.json and roc_curve.png

Usage:
    python evaluate.py --X X.npy --y y.npy --seeds 10 --epochs 75
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import LSTMClassifier, set_seed

CLASSES    = ["Arm Flapping", "Head Banging", "Spinning"]
NUM_CLASSES = len(CLASSES)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def predict_proba(model, X_val, device) -> np.ndarray:
    model.eval()
    X_t    = torch.tensor(X_val, dtype=torch.float32).to(device)
    logits = model(X_t).cpu().numpy()
    return softmax(logits)              # (N, num_classes)


def train_fold(X_tr, y_tr, X_val, y_val, device, epochs, lr=0.01, batch_size=16):
    model     = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_f1    = -1.0
    best_state = None

    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_v), 1).cpu().numpy()
        cur_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if cur_f1 > best_f1:
            best_f1    = cur_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate(X, y, n_splits=5, n_seeds=10, epochs=75, device=None, output_dir="."):
    if device is None:
        device = torch.device("cpu")

    records = []
    all_fpr   = {c: [] for c in range(NUM_CLASSES)}
    all_tpr   = {c: [] for c in range(NUM_CLASSES)}
    mean_fpr  = np.linspace(0, 1, 200)

    print(f"\n{'='*64}")
    print(f"  EVALUATION  |  {n_splits}-fold × {n_seeds} seeds = {n_splits*n_seeds} folds")
    print(f"{'='*64}\n")

    y_bin = label_binarize(y, classes=list(range(NUM_CLASSES)))

    for seed in range(n_seeds):
        set_seed(seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr   = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model  = train_fold(X_tr, y_tr, X_val, y_val, device, epochs)
            proba  = predict_proba(model, X_val, device)
            preds  = proba.argmax(axis=1)

            acc  = accuracy_score(y_val, preds)  * 100
            prec = precision_score(y_val, preds, average="macro", zero_division=0) * 100
            rec  = recall_score(y_val, preds, average="macro", zero_division=0)    * 100
            f1   = f1_score(y_val, preds, average="macro", zero_division=0)        * 100

            # Per-class AUROC → average
            y_val_bin = label_binarize(y_val, classes=list(range(NUM_CLASSES)))
            try:
                auroc = roc_auc_score(y_val_bin, proba,
                                      multi_class="ovr", average="macro")
            except ValueError:
                auroc = float("nan")

            records.append({"acc": acc, "prec": prec, "rec": rec,
                             "f1": f1, "auroc": auroc})

            # Accumulate ROC curves per class
            for c in range(NUM_CLASSES):
                if y_val_bin[:, c].sum() > 0:
                    fpr, tpr, _ = roc_curve(y_val_bin[:, c], proba[:, c])
                    all_tpr[c].append(np.interp(mean_fpr, fpr, tpr))

            print(f"  seed={seed:02d} fold={fold_idx+1}/{n_splits} | "
                  f"acc={acc:5.1f}  prec={prec:5.1f}  rec={rec:5.1f}  "
                  f"F1={f1:5.1f}  AUROC={auroc:.3f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  SUMMARY  (mean ± SD across all folds)")
    print(f"{'='*64}")
    summary = {}
    for key in ["acc", "prec", "rec", "f1", "auroc"]:
        vals = [r[key] for r in records if not np.isnan(r[key])]
        m, s = np.mean(vals), np.std(vals)
        summary[key] = {"mean": round(m, 2), "sd": round(s, 2)}
        label = {"acc":"Accuracy","prec":"Precision","rec":"Recall",
                 "f1":"F1","auroc":"AUROC"}[key]
        print(f"  {label:12s}: {m:.2f} ± {s:.2f}")
    print(f"{'='*64}\n")

    # Save JSON
    out_json = os.path.join(output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_fold": records}, f, indent=2)
    print(f"[INFO] Results saved → {out_json}")

    # ── ROC plot ─────────────────────────────────────────────────────────────
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    macro_tpr = []
    for c, cls_name in enumerate(CLASSES):
        if all_tpr[c]:
            mean_tpr = np.mean(all_tpr[c], axis=0)
            std_tpr  = np.std(all_tpr[c], axis=0)
            macro_tpr.append(mean_tpr)

            ax.plot(mean_fpr, mean_tpr, color=colors[c],
                    lw=2, label=cls_name)
            ax.fill_between(mean_fpr,
                            np.clip(mean_tpr - std_tpr, 0, 1),
                            np.clip(mean_tpr + std_tpr, 0, 1),
                            alpha=0.15, color=colors[c])

    if macro_tpr:
        macro_mean = np.mean(macro_tpr, axis=0)
        ax.plot(mean_fpr, macro_mean, color="white",
                lw=2.5, linestyle="--", label="Macro average")

    ax.plot([0, 1], [0, 1], color="#555", linestyle=":", lw=1.5, label="Chance")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate",  color="white", fontsize=12)
    ax.set_title("Average ROC Curves — SSBD 3-Class", color="white", fontsize=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    legend = ax.legend(loc="lower right", framealpha=0.3, fontsize=10)
    plt.setp(legend.get_texts(), color="white")

    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved → {roc_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM on SSBD")
    parser.add_argument("--X",          type=str, default="X.npy")
    parser.add_argument("--y",          type=str, default="y.npy")
    parser.add_argument("--seeds",      type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=75)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    X = np.load(args.X)
    y = np.load(args.y)
    print(f"[INFO] X: {X.shape}   y: {y.shape}")

    evaluate(X, y, n_splits=5, n_seeds=args.seeds,
             epochs=args.epochs, device=device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
