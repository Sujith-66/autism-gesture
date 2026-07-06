"""
extract_features.py
--------------------
Paper Reference: Lakkapragada et al., JMIR Biomed Eng 2022;7(1):e33771

Step 1 of the pipeline:
- Loads videos from the SSBD dataset (arm_flapping/, head_banging/, spinning/)
- Extracts the first 90 frames per video (as in the paper)
- Passes each frame through MobileNetV2's penultimate layer → 1280-dim feature vector
- Saves X.npy (shape: N x 90 x 1280) and y.npy (shape: N,)

Dataset folder structure expected:
    data/
        arm_flapping/   -> label 0
        head_banging/   -> label 1
        spinning/       -> label 2

Usage:
    python extract_features.py --data_dir ./data --output_dir .
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
NUM_FRAMES   = 90          # Paper: "We took the first 90 frames of a video"
FEATURE_DIM  = 1280        # MobileNetV2 penultimate layer output
IMG_SIZE     = 224
CLASS_MAP    = {
    "arm_flapping": 0,
    "head_banging":  1,
    "spinning":      2,
}

# ── Image transform (ImageNet normalisation) ─────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def build_feature_extractor(device: torch.device) -> nn.Module:
    """
    MobileNetV2 with classifier head replaced by Identity.
    Penultimate layer output = 1280-dim vector.
    Pre-trained on ImageNet; later fine-tuned during LSTM training.
    """
    weights = MobileNet_V2_Weights.DEFAULT
    model   = mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()          # strip the final FC
    model = model.to(device)
    model.eval()
    return model


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """
    Read the first `num_frames` frames from a video file.
    Returns a list of PIL Images (RGB).
    """
    cap    = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


@torch.no_grad()
def frames_to_feature_sequence(frames, model, device) -> np.ndarray:
    """
    Convert a list of PIL frames → (T, 1280) numpy array.
    Frames shorter than NUM_FRAMES are zero-padded.
    """
    seq = np.zeros((NUM_FRAMES, FEATURE_DIM), dtype=np.float32)

    for i, frame in enumerate(frames[:NUM_FRAMES]):
        tensor = transform(frame).unsqueeze(0).to(device)   # (1, 3, 224, 224)
        feat   = model(tensor).cpu().numpy().flatten()       # (1280,)
        seq[i] = feat

    return seq


def process_dataset(data_dir: str, device: torch.device):
    """
    Walk the class folders, extract features, return X and y arrays.
    """
    model   = build_feature_extractor(device)
    X_list, y_list = [], []
    skipped = 0

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Missing folder: {class_dir}")
            continue

        video_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        print(f"\n[{class_name}] Found {len(video_files)} videos (label={label})")

        for vf in tqdm(video_files, desc=class_name):
            vpath  = os.path.join(class_dir, vf)
            frames = extract_frames(vpath)

            if len(frames) < 10:            # skip very short / corrupt clips
                print(f"  [SKIP] {vf}: only {len(frames)} frames")
                skipped += 1
                continue

            seq = frames_to_feature_sequence(frames, model, device)
            X_list.append(seq)
            y_list.append(label)

    print(f"\n[INFO] Total videos processed : {len(X_list)}")
    print(f"[INFO] Total videos skipped   : {skipped}")

    X = np.stack(X_list)           # (N, 90, 1280)
    y = np.array(y_list)           # (N,)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="SSBD Feature Extraction")
    parser.add_argument("--data_dir",   type=str, default="./data",
                        help="Root folder with arm_flapping/, head_banging/, spinning/")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Where to save X.npy and y.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    X, y = process_dataset(args.data_dir, device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "X.npy"), X)
    np.save(os.path.join(args.output_dir, "y.npy"), y)

    print(f"\n[DONE] Saved X.npy {X.shape}  y.npy {y.shape}")
    print(f"       Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")


if __name__ == "__main__":
    main()
