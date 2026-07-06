# ASD Behavior Detection — SSBD Prototype

Implementation of the architecture from:

> **Lakkapragada et al.** *The Classification of Abnormal Hand Movement to Aid in Autism Detection: Machine Learning Study.*  
> JMIR Biomed Eng 2022;7(1):e33771. https://doi.org/10.2196/33771

---

## Architecture (exactly as in paper)

```
Video frames (first 90)
     ↓  [MobileNetV2 penultimate layer]
Feature sequence  (90 × 1280)
     ↓  [LSTM — hidden_size=64]
     ↓  [Dropout 30%]
     ↓  [Linear → 3 classes]
Prediction: Arm Flapping | Head Banging | Spinning
```

| Hyper-parameter  | Value                  |
|------------------|------------------------|
| Frames per video | 90                     |
| Feature dim      | 1 280 (MobileNetV2)    |
| LSTM hidden      | 64                     |
| Dropout          | 30%                    |
| Optimizer        | Adam lr = 0.01         |
| Epochs           | 75                     |
| Evaluation       | 5-fold CV × 100 seeds  |
| Paper test F1    | **84.0 ± 3.7**         |

---

## Project Structure

```
ssbd_prototype/
├── extract_features.py   # Step 1 — MobileNetV2 feature extraction
├── train.py              # Step 2 — LSTM training + k-fold CV
├── evaluate.py           # Step 3 — Full metrics + ROC curves
├── inference.py          # Reusable inference engine
├── app.py                # Flask web application
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Recommended) GPU with CUDA — install PyTorch with CUDA from:
#    https://pytorch.org/get-started/locally/
```

---

## Dataset

Download the **SSBD (Self-Stimulatory Behavior Dataset)** from:  
https://rolandgoecke.net/research/datasets/ssbd/

Organize your downloaded/clipped videos like this:

```
data/
├── arm_flapping/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
├── head_banging/
│   ├── clip_001.mp4
│   └── ...
└── spinning/
    ├── clip_001.mp4
    └── ...
```

The paper used **100 positive + 100 control clips (2–5 seconds each)**.  
For 3-class, aim for ~100 clips per class.

---

## Step-by-Step Usage

### Step 1 — Extract MobileNetV2 Features

```bash
python extract_features.py --data_dir ./data --output_dir .
```

Output: `X.npy` (N × 90 × 1280) and `y.npy` (N,)

---

### Step 2 — Train the LSTM

```bash
# Quick run (10 seeds — ~10 min on CPU)
python train.py --X X.npy --y y.npy --seeds 10 --epochs 75

# Full paper replication (100 seeds — matches paper exactly)
python train.py --X X.npy --y y.npy --seeds 100 --epochs 75

# Just train final model, skip CV
python train.py --X X.npy --y y.npy --skip_cv
```

Output: `lstm_model.pth` + cross-validation metrics in terminal.

---

### Step 3 — Evaluate with Full Metrics + ROC Curves

```bash
python evaluate.py --X X.npy --y y.npy --seeds 10 --output_dir .
```

Output: `results.json` + `roc_curve.png`

---

### Step 4 — Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** — upload a video, get instant prediction.

---

## Expected Results (from paper)

| Approach              | Test F1 (SD)     | AUROC    |
|-----------------------|-----------------|---------|
| All 21 landmarks      | 66.6 (3.35)     | 0.748   |
| 6 landmarks           | 68.3 (3.6)      | 0.760   |
| 1 landmark            | 64.9 (6.5)      | 0.751   |
| Mean landmark         | 64.2 (6.8)      | 0.730   |
| **MobileNetV2 + LSTM**| **84.0 (3.7)**  | **0.85**|

---

## Known Bugs Fixed From Original Code

| File           | Bug                                              | Fix                          |
|----------------|--------------------------------------------------|------------------------------|
| `app.py` (old) | `labels` has 3 items but `probs[3]` accessed     | Removed 4th label reference  |
| `app.py` (old) | LSTM `fc` outputs 3 but HTML showed 4 classes    | Consistent 3-class design    |
| `train_lstm.py`| `fc = Linear(128, 3)` — wrong hidden size        | Paper uses 64, not 128       |
| `train_lstm.py`| lr=0.001                                         | Paper uses lr=0.01           |
| `app.py` (old) | `model_path="../lstm_model.pth"` hardcoded       | Configurable via constructor |

---

## Citation

```bibtex
@article{lakkapragada2022,
  title   = {The Classification of Abnormal Hand Movement to Aid
             in Autism Detection: Machine Learning Study},
  author  = {Lakkapragada, A and Kline, A and Mutlu, OC and
             Paskov, K and Chrisman, B and Stockham, N and
             Washington, P and Wall, DP},
  journal = {JMIR Biomed Eng},
  year    = {2022},
  volume  = {7},
  number  = {1},
  pages   = {e33771},
  doi     = {10.2196/33771}
}
```
