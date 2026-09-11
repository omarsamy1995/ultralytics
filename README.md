<p align="center">
  <img src="assets/banner.jpg" alt="KLYVERO Engine Banner" width="100%">
</p>

<p align="center">
  <b>Custom 12-Keypoint Body Pose Estimation · CBAM Attention · Temporal Smoothing · Virtual Try-On Ready</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-v11-blue?style=for-the-badge&logo=yolo" alt="YOLOv11">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-orange?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Task-Pose_Estimation-purple?style=for-the-badge" alt="Task">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Model Comparison](#model-comparison)
- [Quick Start](#quick-start)
- [Real-Time Inference](#real-time-inference)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

**KLYVERO Engine** is a custom-trained YOLO11 pose estimation model designed for **virtual try-on** and **fashion technology** applications. Unlike standard COCO-based pose models that detect 17 keypoints (including facial landmarks), KLYVERO uses an optimized **12-keypoint body skeleton** focused exclusively on garment-relevant anatomical landmarks.

The model incorporates two key architectural innovations:
1. **CBAM (Convolutional Block Attention Module)** — channel + spatial attention injected after the backbone for superior feature refinement
2. **Temporal Consistency Loss** — a custom loss function that penalizes frame-to-frame keypoint jitter, producing smoother predictions in video sequences

### Why 12 Keypoints?

| Standard COCO (17 kpt) | KLYVERO (12 kpt) |
|:-:|:-:|
| Nose, Eyes, Ears + Body | Body-only landmarks |
| Redundant facial points for try-on | Every keypoint maps to a garment anchor |
| Generic purpose | Fashion / try-on optimized |

The 12 keypoints are:

```
Left Shoulder ↔ Right Shoulder
Left Elbow    ↔ Right Elbow
Left Wrist    ↔ Right Wrist
Left Hip      ↔ Right Hip
Left Knee     ↔ Right Knee
Left Ankle    ↔ Right Ankle
```

---

## Key Innovations

### 🔬 CBAM Attention Module

The **Convolutional Block Attention Module** is inserted after the C2PSA layer in the YOLO11 backbone, enabling the model to focus on the most relevant spatial regions and channel features for pose estimation.

```
Input → Channel Attention (squeeze → excite) → Spatial Attention (pool → conv) → Output
```

This helps the model:
- Better distinguish body parts in cluttered backgrounds
- Improve keypoint localization accuracy on occluded joints
- Maintain high precision across varied clothing types and poses

### ⏱️ Temporal Consistency Loss

A custom loss component that minimizes the L2 distance between predicted keypoints across consecutive frames during training. This produces:
- **Smoother** keypoint trajectories in video
- **Reduced jitter** without any post-processing
- **Production-ready** stability for real-time try-on applications

### 🔄 EMA Temporal Smoothing (Inference)

At inference time, an Exponential Moving Average (EMA) filter further stabilizes keypoints:

```python
smoothed = α × current + (1 - α) × previous   # α = 0.5
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOLO11n-CBAM-Pose                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BACKBONE                                                       │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │Conv  │→│Conv  │→│C3k2  │→│Conv  │→│C3k2  │→│Conv  │  │
│  │P1/2  │  │P2/4  │  │      │  │P3/8  │  │      │  │P4/16 │  │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  │
│       ↓                                                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────────────────┐         │
│  │C3k2  │→│Conv  │→│C3k2  │→│ SPPF → C2PSA       │         │
│  │      │  │P5/32 │  │      │  │   → ★ CBAM ★     │         │
│  └──────┘  └──────┘  └──────┘  └────────────────────┘         │
│                                                                 │
│  HEAD (FPN + PAN)                                               │
│  ┌──────────────────────────────────────────────┐               │
│  │  Upsample → Concat(P4) → C3k2               │               │
│  │  Upsample → Concat(P3) → C3k2  [P3/8]       │               │
│  │  Downsample → Concat    → C3k2  [P4/16]      │               │
│  │  Downsample → Concat    → C3k2  [P5/32]      │               │
│  └──────────────────────────────────────────────┘               │
│       ↓                                                         │
│  ┌──────────────────────────┐                                   │
│  │  Pose Head               │                                   │
│  │  1 class × 12 keypoints  │                                   │
│  │  (x, y, visibility) × 12 │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Model specs (nano scale):**
- **Layers:** 196
- **Parameters:** ~2.9M
- **GFLOPs:** 7.7
- **Input size:** 640 × 640

---

## Dataset

The **KLYVERO Dataset** is a curated collection of fashion and yoga pose images with 12-keypoint body annotations.

| Split | Images | Purpose |
|-------|-------:|---------|
| Train | 2,876 | Model training with augmentation |
| Val | 667 | Validation & metric tracking |
| Test | 334 | Final evaluation |
| **Total** | **3,877** | |

**Annotation format:** YOLO pose format with `[class, x, y, w, h, kp1_x, kp1_y, kp1_v, ..., kp12_x, kp12_y, kp12_v]`

<p align="center">
  <img src="assets/results/labels.jpg" alt="Dataset label distribution and bounding box analysis" width="700">
</p>
<p align="center"><i>Dataset statistics: class distribution, bounding box positions and dimensions</i></p>

---

## Training Pipeline

The model was trained through a **3-stage progressive pipeline**, each building upon the previous stage:

### Stage 1 — Baseline (YOLO11n-Pose)
> Transfer learning from COCO-pretrained YOLO11n-pose with first 10 layers frozen

| Parameter | Value |
|-----------|-------|
| Base model | `yolo11n-pose.pt` (COCO pretrained) |
| Epochs | 300 |
| Patience | 50 |
| Batch size | 16 |
| Image size | 640 |
| Optimizer | Auto (AdamW) |
| Frozen layers | 10 |
| Augmentations | Mosaic, RandAugment, HSV, Flip, Erasing |

### Stage 2 — CBAM + Temporal Loss
> Injected CBAM attention into backbone + added temporal consistency loss

| Parameter | Value |
|-----------|-------|
| Architecture | `yolo11-cbam-pose.yaml` |
| Pretrained from | Stage 1 best weights |
| Custom loss | Temporal consistency (L2 penalty) |
| Epochs | 50+ |

### Stage 3 — Final Model (Extended Training)
> Continued training with expanded dataset and refined hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | `yolo11-cbam-pose.yaml` |
| Pretrained from | Stage 2 best weights |
| Epochs | 200+ |
| Learning rate | 0.001 → 0.01 (cosine) |
| Pose loss weight | 12.0 |
| Box loss weight | 7.5 |

---

## Results

### Final Model Performance

<p align="center">
  <img src="assets/results/results_final.png" alt="KLYVERO Final Model — Training curves showing all loss components and metrics over 200 epochs" width="100%">
</p>
<p align="center"><i>Final model training curves: all losses converge smoothly, metrics plateau near optimal values</i></p>

#### Detection Metrics (Bounding Box)

| Metric | Value |
|--------|------:|
| Precision | **99.97%** |
| Recall | **100.00%** |
| mAP@50 | **99.50%** |
| mAP@50-95 | **89.49%** |

#### Pose Estimation Metrics (Keypoints)

| Metric | Value |
|--------|------:|
| Precision | **95.78%** |
| Recall | **95.80%** |
| mAP@50 | **93.27%** |
| mAP@50-95 | **78.88%** |

---

### Confusion Matrix

<p align="center">
  <img src="assets/results/confusion_matrix.png" alt="Confusion matrix showing 667 correct predictions and only 2 false positives" width="500">
</p>
<p align="center"><i>Near-perfect detection: 667 true positives, 0 false negatives, only 2 false positives</i></p>

---

### Precision-Recall & F1 Curves

<p align="center">
  <table>
    <tr>
      <td><img src="assets/results/PosePR_curve.png" alt="Pose Precision-Recall curve showing 0.945 mAP@0.5" width="400"></td>
      <td><img src="assets/results/PoseF1_curve.png" alt="Pose F1-Confidence curve showing 0.96 peak F1 at confidence 0.571" width="400"></td>
    </tr>
    <tr>
      <td align="center"><i>Pose PR Curve — mAP@0.5 = 0.945</i></td>
      <td align="center"><i>Pose F1 Curve — Peak F1 = 0.96 @ conf 0.571</i></td>
    </tr>
  </table>
</p>

<p align="center">
  <table>
    <tr>
      <td><img src="assets/results/BoxPR_curve.png" alt="Box Precision-Recall curve" width="400"></td>
      <td><img src="assets/results/BoxF1_curve.png" alt="Box F1-Confidence curve" width="400"></td>
    </tr>
    <tr>
      <td align="center"><i>Box PR Curve</i></td>
      <td align="center"><i>Box F1 Curve</i></td>
    </tr>
  </table>
</p>

---

### Visual Predictions

<p align="center">
  <img src="assets/results/val_batch0_pred.jpg" alt="Validation predictions showing accurate 12-keypoint pose estimation on yoga poses" width="700">
</p>
<p align="center"><i>Validation predictions — Yoga poses with accurate 12-keypoint detection (confidence ≥ 0.8)</i></p>

<p align="center">
  <img src="assets/results/val_batch1_pred.jpg" alt="Validation predictions on fashion catalog images with front, back, and side poses" width="700">
</p>
<p align="center"><i>Validation predictions — Fashion catalog images across front, back, and side views</i></p>

---

### Training Samples

<p align="center">
  <img src="assets/results/train_batch0.jpg" alt="Training batch with augmented samples showing mosaic, color jitter, and flipping" width="700">
</p>
<p align="center"><i>Training batch with augmentations: mosaic, HSV jitter, random erasing, horizontal flip</i></p>

---

## Model Comparison

### 3-Stage Training Progression

| Stage | Model | Epochs | Box mAP@50 | Box mAP@50-95 | Pose mAP@50 | Pose mAP@50-95 |
|:---:|-------|:---:|:---:|:---:|:---:|:---:|
| 1 | YOLO11n-Pose (Baseline) | 300 | 99.49% | 89.12% | 93.35% | 80.19% |
| 2 | + CBAM + Temporal Loss | 50+ | 99.48% | 89.00% | 93.45% | 79.90% |
| 3 | **Final (Extended)** | 200+ | **99.50%** | **89.49%** | **93.41%** | 79.24% |

<p align="center">
  <table>
    <tr>
      <td><img src="assets/results/results_baseline.png" alt="Stage 1 baseline training curves" width="100%"></td>
    </tr>
    <tr>
      <td align="center"><i>Stage 1 — Baseline YOLO11n-Pose (300 epochs)</i></td>
    </tr>
    <tr>
      <td><img src="assets/results/results_cbam_temporal.png" alt="Stage 2 CBAM + temporal loss training curves" width="100%"></td>
    </tr>
    <tr>
      <td align="center"><i>Stage 2 — CBAM + Temporal Loss (rapid convergence in ~50 epochs)</i></td>
    </tr>
    <tr>
      <td><img src="assets/results/results_final.png" alt="Stage 3 final model training curves" width="100%"></td>
    </tr>
    <tr>
      <td align="center"><i>Stage 3 — Final Model with expanded dataset (200+ epochs)</i></td>
    </tr>
  </table>
</p>

**Key takeaways:**
- CBAM attention enabled **faster convergence** (Stage 2 achieved competitive metrics in ~50 epochs vs 300)
- Temporal loss introduced smooth loss curves with **near-zero temporal loss** by convergence
- The final model achieves **near-perfect detection** (100% recall) with strong keypoint accuracy

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/KLYVERO_Advanced_YOLO.git
cd KLYVERO_Advanced_YOLO

# Install dependencies
pip install ultralytics opencv-python numpy
```

### Inference

```python
from ultralytics import YOLO

# Load the KLYVERO final model
model = YOLO('ultralytics/runs/pose/KLYVERO_Engine/KLYVERO_Final_Model/weights/best.pt')

# Run inference on an image
results = model('your_image.jpg', conf=0.5)

# Access keypoints
for r in results:
    keypoints = r.keypoints.xy[0]  # Shape: [12, 2] — 12 body keypoints
    print(keypoints)
```

### Keypoint Index Mapping

```python
KEYPOINTS = {
    0:  'left_shoulder',   1:  'right_shoulder',
    2:  'left_elbow',      3:  'right_elbow',
    4:  'left_wrist',      5:  'right_wrist',
    6:  'left_hip',        7:  'right_hip',
    8:  'left_knee',       9:  'right_knee',
    10: 'left_ankle',      11: 'right_ankle',
}
```

---

## Real-Time Inference

The repository includes a real-time webcam inference script with **temporal smoothing** for production-quality keypoint stability:

```bash
python ultralytics/test-live.py
```

This script:
1. Loads the KLYVERO final model
2. Captures webcam frames in real-time
3. Applies **EMA temporal smoothing** (α = 0.5) to keypoints
4. Renders smoothed keypoints as colored dots on the video feed

Press `q` to quit.

### Smoothing Parameter Guide

| Alpha (α) | Behavior | Best For |
|:-:|----------|----------|
| 0.1 | Heavy smoothing, slow response | Static try-on displays |
| 0.3 | Moderate smoothing | Controlled environments |
| **0.5** | **Balanced (default)** | **Virtual try-on** |
| 0.7 | Light smoothing, fast response | Active movement tracking |
| 0.9 | Minimal smoothing | Sports / dance analysis |

---

## Project Structure

```
KLYVERO_Advanced_YOLO/
├── assets/                          # README images & banner
│   ├── banner.jpg
│   └── results/                     # Training result visualizations
├── README.md                        # This file
├── yolo11n-pose.pt                  # Base COCO pretrained model
│
└── ultralytics/                     # Modified Ultralytics framework
    ├── KLYVERO_Dataset/             # Custom dataset
    │   ├── data.yaml                # Dataset config (12 kpt, 1 class)
    │   ├── train/                   # 2,876 training images + labels
    │   ├── val/                     # 667 validation images + labels
    │   └── test/                    # 334 test images + labels
    │
    ├── ultralytics/
    │   ├── cfg/models/11/
    │   │   └── yolo11-cbam-pose.yaml   # ★ Custom CBAM architecture
    │   ├── nn/modules/
    │   │   ├── conv.py              # CBAM attention implementation
    │   │   └── block.py             # CBAM block variant
    │   └── utils/
    │       └── loss.py              # Temporal consistency loss
    │
    ├── runs/pose/KLYVERO_Engine/    # All training runs
    │   ├── yolo11n_body_12kpt-2/    # Stage 1: Baseline
    │   ├── yolo11n_CBAM_Temporal_Loss-8/  # Stage 2: CBAM + Temporal
    │   └── KLYVERO_Final_Model/     # Stage 3: Final model ★
    │       ├── weights/
    │       │   ├── best.pt          # ★ Best model weights
    │       │   └── last.pt
    │       ├── results.png
    │       ├── confusion_matrix.png
    │       └── ...
    │
    ├── test-live.py                 # Real-time webcam inference
    ├── auto_annotate.py             # Auto-annotation utility
    ├── train_multitask.py           # CBAM training script
    └── train_temporal.ipynb         # Temporal loss training notebook
```

---

## Training (Reproduce)

### Stage 1 — Baseline

```python
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
model.train(
    data='ultralytics/KLYVERO_Dataset/data.yaml',
    epochs=300,
    patience=50,
    imgsz=640,
    batch=16,
    freeze=10,
    project='KLYVERO_Engine',
    name='baseline_12kpt',
)
```

### Stage 2 — CBAM + Temporal Loss

```python
from ultralytics import YOLO

model = YOLO('ultralytics/ultralytics/cfg/models/11/yolo11-cbam-pose.yaml')
model.train(
    data='ultralytics/KLYVERO_Dataset/data.yaml',
    pretrained='runs/pose/KLYVERO_Engine/baseline_12kpt/weights/best.pt',
    epochs=100,
    imgsz=640,
    batch=16,
    freeze=10,
    project='KLYVERO_Engine',
    name='cbam_temporal',
)
```

### Stage 3 — Final Model

```python
from ultralytics import YOLO

model = YOLO('ultralytics/ultralytics/cfg/models/11/yolo11-cbam-pose.yaml')
model.train(
    data='ultralytics/KLYVERO_Dataset/data.yaml',
    pretrained='runs/pose/KLYVERO_Engine/cbam_temporal/weights/best.pt',
    epochs=300,
    patience=30,
    imgsz=640,
    batch=16,
    freeze=10,
    pose=12.0,     # Higher pose loss weight
    box=7.5,
    project='KLYVERO_Engine',
    name='final_model',
)
```

---

## Auto-Annotation Tool

Quickly generate 12-keypoint labels for new images using the trained model:

```python
from ultralytics import YOLO

model = YOLO('path/to/KLYVERO_Final_Model/weights/best.pt')
results = model.predict(source='new_images/', stream=True, conf=0.5)

for result in results:
    # Extract keypoints in YOLO format
    keypoints = result.keypoints.xyn[0]  # Normalized coordinates
    # Save as label files...
```

See [`auto_annotate.py`](ultralytics/auto_annotate.py) for the complete implementation.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU (4GB VRAM) | NVIDIA T4 / RTX 3060+ (16GB) |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 5 GB |
| CUDA | 11.7+ | 12.0+ |

Training was performed on **NVIDIA T4 GPU** via Lightning.ai cloud.

---

## Citation

If you use KLYVERO Engine in your research or application, please cite:

```bibtex
@software{klyvero_engine_2026,
  title   = {KLYVERO Engine: Advanced YOLO11 Body Pose Estimation with CBAM Attention},
  author  = {KLYVERO Team},
  year    = {2026},
  url     = {https://github.com/your-username/KLYVERO_Advanced_YOLO},
  note    = {Custom 12-keypoint pose estimation for virtual try-on applications}
}
```

---

## License

This project is built upon [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and is licensed under the [AGPL-3.0 License](ultralytics/LICENSE).

---

<p align="center">
  <b>Built with ❤️ by KLYVERO Team</b><br>
  <sub>Powering the future of virtual try-on technology</sub>
</p>
