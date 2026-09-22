# Explainable Vision Transformers (ViT) for Multi-Class Skin Lesion Classification

An end-to-end, clinically viable, and transparent deep-learning framework designed to address the **"Black Box" dilemma** and **CNN architectural limitations** in dermatological AI. 

The pipeline combines a **Vision Transformer (ViT)** with **Attention Rollout Explainable AI (XAI)**, multi-class imbalance mitigation (Focal Loss), and an interactive clinical decision-support web application.

---

## 🌟 Key Innovations & Clinical Advantages

1. **Global Spatial Attention over Local Convolutions:**
   - CNNs process images via small localized kernels ($3 \times 3$), struggling with asymmetrical, expansive lesion borders without deep, computationally heavy layers.
   - ViT divides dermoscopic images into non-overlapping $16 \times 16$ patches and leverages **Multi-Head Self-Attention** to map long-range spatial correlations across the entire lesion surface simultaneously.

2. **Full-Depth Visual Explainability via Attention Rollout:**
   - Solves the black-box dilemma by recursively multiplying self-attention matrices across all transformer layers (accounting for residual skip connections $\hat{A}_l = 0.5 A_l + 0.5 I$).
   - Computes explicit patch-level information flow from input tokens to the `[CLS]` classification token, generating high-resolution heatmaps overlaid directly on the dermoscopy image.

3. **Immunity to the "Clever Hans" Artifact Trap:**
   - CNNs often artificially inflate accuracy by attending to peripheral artifacts (dark hairs, surgical ink markings, dermoscopy gel bubbles).
   - ViT's global context focuses on authentic tumor cellular pathology and irregular border patterns rather than isolated background artifacts.

4. **Multi-Class Imbalance Handling:**
   - Implements **Focal Loss** and **Class-Balanced Loss** based on effective number of samples to counter severe imbalance in datasets like ISIC 2024 / HAM10000.

---

## 🏗️ Project Architecture

```
cap_proj/
├── app.py                      # Interactive Streamlit Clinical Decision Support App
├── train.py                    # Training script with dataset loaders & CLI arguments
├── test_pipeline.py            # Automated test suite and validation pipeline
├── checkpoints/                # Saved best model checkpoints (.pt)
├── data/
│   └── demo_samples/           # Pre-generated clinical demo dermoscopy dataset & metadata
└── src/
    ├── models/
    │   ├── vit_classifier.py   # ExplainableViT with attention weight retention hooks
    │   ├── cnn_baseline.py     # ResNet/ConvNet baseline with Grad-CAM gradient hooks
    │   └── loss_functions.py   # Focal Loss & Class-Balanced Loss for imbalance
    ├── xai/
    │   ├── attention_rollout.py # Multi-layer Attention Rollout algorithm
    │   ├── gradcam.py          # Baseline CNN Grad-CAM implementation
    │   └── visualizer.py       # Heatmap blending, colormaps & artifact detection
    ├── data/
    │   ├── dataset.py          # PyTorch Dataset loader for ISIC/HAM10000
    │   ├── transforms.py       # Clinical augmentations & hair artifact simulator
    │   └── sample_data.py      # Realistic dermoscopy case generator
    ├── eval/
    │   ├── metrics.py          # Clinical metrics (Balanced Acc, Mel-Sensitivity, OvR ROC-AUC)
    │   └── benchmark.py        # ViT vs CNN architectural & explainability benchmark
    └── training/
        └── trainer.py          # Training loop with AMP, Cosine Annealing, and Early Stopping
```

---

## 🚀 Quick Start Guide

### 1. Launch the Clinical Decision Support Web Application
Run the FastAPI application server:
```bash
python server.py
```
*(or `uvicorn server:app --reload --port 8000`)*

Then open your browser at **`http://localhost:8000`** (or `http://127.0.0.1:8000`).

The interactive Clinical Decision Support interface allows you to:
- **Upload dermoscopy images** (JPG/PNG) or pick from a preloaded library of clinical cases across 7 diagnostic categories.
- **View Real-Time Attention Rollout Heatmaps** with dynamic colormaps (`turbo`, `jet`, `viridis`, `plasma`, `inferno`), opacity controls, and noise discard ratio tuning.
- **Inspect Malignancy Risk Stratification** (High Risk for Melanoma/BCC, Moderate for Actinic Keratosis, Benign for Nevi).
- **Run Architectural Benchmarks** comparing ViT Attention Rollout against CNN Grad-CAM on cases with hair and marker artifacts.
- **Explore Layer-by-Layer Attention Depth** to visualize feature abstraction from shallow to deep layers.

---

### 2. Train on ISIC 2024 Challenge (3D-TBP) / Custom Datasets
To train the **Hybrid CNN-ViT (Paper 02)** or **Pretrained Vision Transformer (Papers 01, 04, 05)** on the official **ISIC 2024 3D Total Body Photography Dataset**:

```bash
# 1. Train Hybrid CNN-ViT with Focal Loss (Paper 02)
python train_isic2024.py --model_type hybrid_vit --epochs 25 --lr 0.0002 --loss_type focal

# 2. Train Pretrained ImageNet ViT Backbone (Papers 01, 05)
python train_isic2024.py --model_type pretrained_vit --epochs 20 --lr 0.0001 --loss_type focal

# 3. Train on external ISIC 2024 HDF5 archive and metadata
python train_isic2024.py --data_dir /path/to/isic2024 --csv_file /path/to/train-metadata.csv --hdf5_file /path/to/train-image.hdf5 --model_type hybrid_vit
```

The script evaluates official **Partial ROC-AUC above 80% TPR ($p\text{AUC}_{80}$)**, Balanced Accuracy, and Melanoma Sensitivity, and automatically saves top checkpoints to `checkpoints/vit_lesion_classifier_best.pt`.

---

### 3. Run Automated Validation Suites
Verify the entire medical AI and ISIC 2024 pipeline:
```bash
# Test FastAPI Web Server & Endpoints
python test_server.py

# Test ISIC 2024 Ingestion & HybridViT Attention Rollout
python test_isic2024.py
```

---

## 📊 Supported Diagnostic Classes (ISIC Standard)

| Class Code | Full Diagnostic Name | Clinical Risk Level |
| :--- | :--- | :--- |
| **MEL** | Malignant Melanoma | 🔴 **High Risk** (Urgent biopsy/excision) |
| **BCC** | Basal Cell Carcinoma | 🔴 **High Risk** (Surgical excision / Mohs) |
| **AKIEC** | Actinic Keratosis / Bowen's | 🟡 **Moderate Risk** (Pre-cancerous) |
| **NV** | Melanocytic Nevus (Mole) | 🟢 **Benign** (Routine monitoring) |
| **BKL** | Benign Keratosis (Seborrheic) | 🟢 **Benign** (Reassurance) |
| **DF** | Dermatofibroma | 🟢 **Benign** (Fibrous nodule) |
| **VASC** | Vascular Lesion (Angioma) | 🟢 **Benign** (Vascular lacunes) |
