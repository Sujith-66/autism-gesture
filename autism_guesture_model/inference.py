"""
inference.py
-------------
Reusable inference engine for SSBD behavior detection.
Called by app.py to process uploaded videos.

Pipeline per video:
  1. Extract first 90 frames (paper specification)
  2. Pass each frame through MobileNetV2 → 1280-dim feature vector
  3. Stack into sequence (90, 1280)
  4. Pass through trained LSTM → class probabilities
  5. Apply confidence threshold → if max prob < CONFIDENCE_THRESHOLD,
     classify as "No Behavior Detected" (normal / no autism indicator)
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
import torch.nn as nn

NUM_FRAMES  = 90
FEATURE_DIM = 1280
IMG_SIZE    = 224

# ── Confidence threshold ─────────────────────────────────────────────────────
# If the model's maximum class probability is below this value, the video is
# considered to show NO autism-related self-stimulatory behavior.
# Rationale: the LSTM was trained only on positive examples (3 behaviors).
# A low max-prob means the video doesn't strongly resemble any of them.
CONFIDENCE_THRESHOLD = 0.55   # 55% — tune this based on your validation set

# ── Class definitions ────────────────────────────────────────────────────────
CLASSES = ["Arm Flapping", "Head Banging", "Spinning"]
COLORS  = ["#E74C3C",      "#F39C12",      "#3498DB"]

# Special "no behavior" constants used by app.py
NO_BEHAVIOR_LABEL = "No Behavior Detected"
NO_BEHAVIOR_COLOR = "#10b981"   # green

# ── Class descriptions used only for the human-readable report ──────────────
# These describe the repetitive-motion pattern each class was TRAINED to
# recognize. They are static reference text, not something computed from the
# clip — the actual evidence for a given prediction is the probability/margin
# data below.
CLASS_MOTION_CUES = {
    "Arm Flapping": "rapid, rhythmic up-and-down or side-to-side motion of the "
                     "hands/forearms, typically with limited motion elsewhere",
    "Head Banging":  "repetitive forward-backward or side-to-side motion of the "
                     "head, often against a relatively still body/torso",
    "Spinning":      "repetitive rotational motion of the whole body or "
                     "head, sustained across consecutive frames",
}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ── Inline LSTM model (no dependency on train.py) ────────────────────────────
class LSTMClassifier(nn.Module):
    """
    Exactly the architecture described in the paper:
      LSTM(64) → Dropout(0.30) → Linear(num_classes)
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
        out, _  = self.lstm(x)        # (B, T, 64)
        out     = out[:, -1, :]       # last timestep → (B, 64)
        out     = self.dropout(out)
        return self.fc(out)           # (B, num_classes)


class BehaviorDetector:
    """
    Wraps MobileNetV2 feature extractor + trained LSTM classifier.
    Thread-safe (models are loaded once, eval mode).

    Confidence-threshold logic:
        max_prob >= CONFIDENCE_THRESHOLD  → one of the 3 ASD behaviors detected
        max_prob <  CONFIDENCE_THRESHOLD  → "No Behavior Detected" (normal)
    """

    def __init__(self, model_path: str = "lstm_model.pth",
                       device: str = "auto",
                       confidence_threshold: float = CONFIDENCE_THRESHOLD):

        self.confidence_threshold = confidence_threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── MobileNetV2 feature extractor ────────────────────────────────────
        weights  = MobileNet_V2_Weights.DEFAULT
        cnn      = mobilenet_v2(weights=weights)
        cnn.classifier = nn.Identity()
        self.cnn = cnn.to(self.device).eval()

        # ── LSTM classifier ──────────────────────────────────────────────────
        lstm = LSTMClassifier(
            input_size  = FEATURE_DIM,
            hidden_size = 64,
            num_classes = len(CLASSES),
            dropout     = 0.0,        # no dropout at inference
        )
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            lstm.load_state_dict(state)
            print(f"[INFO] Loaded model weights from {model_path}")
        else:
            print(f"[WARN] Model weights not found at {model_path}. "
                  f"Using random weights — run train.py first.")

        self.lstm = lstm.to(self.device).eval()

    @torch.no_grad()
    def _extract_cnn_features(self, frame_pil: Image.Image) -> np.ndarray:
        t    = transform(frame_pil).unsqueeze(0).to(self.device)
        feat = self.cnn(t).detach().cpu().numpy().flatten()
        return feat

    def _load_frames(self, video_path: str):
        cap    = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < NUM_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames

    # ── Report builder ────────────────────────────────────────────────────
    # NOTE: this does not change any prediction logic — it only formats a
    # short summary from numbers the model already produced (probabilities,
    # confidence, margin, frame count). No new model calls, no new heavy
    # computation, so the backend pipeline itself is untouched.
    def _build_report(self, probs: np.ndarray, num_frames: int,
                       autism_detected: bool, label: str) -> dict:
        order        = np.argsort(probs)[::-1]            # indices, high → low
        top_idx      = int(order[0])
        second_idx   = int(order[1])
        top_prob     = float(probs[top_idx])
        second_prob  = float(probs[second_idx])
        margin       = top_prob - second_prob              # decision margin

        frame_coverage = f"{num_frames}/{NUM_FRAMES}"
        low_coverage   = num_frames < NUM_FRAMES

        if margin >= 0.30:
            certainty = "High"
        elif margin >= 0.12:
            certainty = "Moderate"
        else:
            certainty = "Low"

        factors = [
            f"Top predicted class \"{CLASSES[top_idx]}\" reached "
            f"{top_prob*100:.1f}% probability, vs. {second_prob*100:.1f}% for the "
            f"next closest class (\"{CLASSES[second_idx]}\") — a decision margin "
            f"of {margin*100:.1f} percentage points.",
            f"Decision was based on {frame_coverage} video frames analyzed "
            f"through the MobileNetV2 + LSTM pipeline."
            + (" Fewer frames than the model's full 90-frame window were "
               "available, which can reduce reliability." if low_coverage else ""),
        ]

        if autism_detected:
            cue = CLASS_MOTION_CUES.get(label, "a repetitive motion pattern")
            headline = (
                f"The clip most closely matches the visual/motion pattern the "
                f"model associates with \"{label}\" ({cue})."
            )
            factors.insert(0, headline)
            summary = (
                f"The model classified this clip as \"{label}\" with "
                f"{top_prob*100:.1f}% confidence (decision certainty: {certainty}). "
                f"This is based on learned visual-motion patterns, not a clinical "
                f"evaluation."
            )
        else:
            summary = (
                f"No class crossed the {self.confidence_threshold*100:.0f}% "
                f"confidence threshold required to flag a behavior — the closest "
                f"match was \"{CLASSES[top_idx]}\" at only {top_prob*100:.1f}%. "
                f"The model treats this as no self-stimulatory pattern detected."
            )

        return {
            "summary":          summary,
            "certainty":        certainty,
            "decision_margin":  round(margin, 4),
            "frame_coverage":   frame_coverage,
            "low_frame_coverage": low_coverage,
            "factors":          factors,
            "disclaimer":       (
                "This is an automated pattern-classification result from a "
                "research prototype trained on a small video dataset (SSBD). "
                "It identifies which of three known stereotypy patterns a clip "
                "most resembles — it does not diagnose Autism Spectrum Disorder "
                "and should never replace evaluation by a qualified clinician."
            ),
        }

    @torch.no_grad()
    def predict(self, video_path: str) -> dict:
        """
        Run the full pipeline on a video file.

        Returns:
            {
              "label":         str,    # predicted class name OR "No Behavior Detected"
              "label_idx":     int,    # 0|1|2 for behaviors, -1 for no behavior
              "probabilities": [float, float, float],  # always the 3 class probs
              "max_confidence": float, # highest class probability (0-1)
              "confidence_threshold": float,
              "autism_detected": bool, # True if a behavior was found
              "num_frames":    int,
              "error":         str | None
            }
        """
        frames = self._load_frames(video_path)

        if len(frames) < 5:
            return {
                "error": f"Video too short ({len(frames)} frames). Need at least 5 frames.",
                "label": None, "label_idx": None,
                "probabilities": None, "max_confidence": None,
                "confidence_threshold": self.confidence_threshold,
                "autism_detected": False, "num_frames": len(frames),
                "report": None,
            }

        # Build (1, 90, 1280) sequence — zero-pad if video < 90 frames
        seq = np.zeros((NUM_FRAMES, FEATURE_DIM), dtype=np.float32)
        for i, frame in enumerate(frames[:NUM_FRAMES]):
            seq[i] = self._extract_cnn_features(frame)

        seq_t  = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.lstm(seq_t)                         # (1, 3)
        probs  = torch.softmax(logits, dim=1).detach().cpu().numpy().flatten()

        max_confidence = float(probs.max())
        label_idx      = int(probs.argmax())

        # ── Confidence gate ───────────────────────────────────────────────────
        if max_confidence < self.confidence_threshold:
            # No behavior was detected with sufficient confidence
            return {
                "label":                NO_BEHAVIOR_LABEL,
                "label_idx":            -1,
                "probabilities":        probs.tolist(),
                "max_confidence":       max_confidence,
                "confidence_threshold": self.confidence_threshold,
                "autism_detected":      False,
                "num_frames":           len(frames),
                "error":                None,
                "report": self._build_report(probs, len(frames),
                                              autism_detected=False,
                                              label=NO_BEHAVIOR_LABEL),
            }

        # ── Behavior detected ─────────────────────────────────────────────────
        return {
            "label":                CLASSES[label_idx],
            "label_idx":            label_idx,
            "probabilities":        probs.tolist(),
            "max_confidence":       max_confidence,
            "confidence_threshold": self.confidence_threshold,
            "autism_detected":      True,
            "num_frames":           len(frames),
            "error":                None,
            "report": self._build_report(probs, len(frames),
                                          autism_detected=True,
                                          label=CLASSES[label_idx]),
        }
