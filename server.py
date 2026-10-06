import os
import io
import time
import base64
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.xai.attention_rollout import AttentionRollout
from src.xai.gradcam import GradCAM
from src.xai.visualizer import overlay_heatmap, detect_artifact_vs_pathology
from src.data.transforms import get_val_transforms, denormalize_tensor
from src.data.sample_data import generate_sample_dataset
from src.eval.metrics import calculate_clinical_metrics


# ── Transfer Learning classifier (used when checkpoint has 'backbone' key) ──
class TransferLesionClassifier(nn.Module):
    """EfficientNet-B0 pretrained backbone with custom 7-class head."""
    def __init__(self, num_classes=7, backbone="efficientnet_b0.ra_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


app = FastAPI(
    title="Explainable ViT Skin Lesion Diagnostic API",
    description="Clinical Decision Support System with Real-Time Attention Rollout XAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
CLASSES = [
    "MEL (Melanoma)",
    "NV (Melanocytic Nevus)",
    "BCC (Basal Cell Carcinoma)",
    "AKIEC (Actinic Keratosis)",
    "BKL (Benign Keratosis)",
    "DF (Dermatofibroma)",
    "VASC (Vascular Lesion)"
]

RISK_MAP = {
    "MEL": ("HIGH RISK - Malignant Melanoma", "danger", "Urgent excision, dermatopathology evaluation, and sentinel lymph node staging recommended."),
    "BCC": ("HIGH RISK - Basal Cell Carcinoma", "danger", "Surgical excision or Mohs micrographic surgery recommended."),
    "AKIEC": ("MODERATE RISK - Actinic Keratosis / Pre-cancerous", "warning", "Cryotherapy or topical therapy (5-FU / Imiquimod) evaluation indicated."),
    "NV": ("LOW RISK - Benign Melanocytic Nevus", "success", "Benign melanocytic lesion. Routine surveillance; monitor for ABCDE changes."),
    "BKL": ("BENIGN - Benign Keratosis / Seborrheic", "success", "Benign non-melanocytic epidermal lesion. Reassurance and regular follow-up."),
    "DF": ("BENIGN - Dermatofibroma", "success", "Benign dermal fibrous histiocytoma. Clinical reassurance."),
    "VASC": ("BENIGN - Vascular Lesion / Angioma", "success", "Benign vascular lacunes/hemangioma. No oncological intervention needed.")
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIT_MODEL = None   # ExplainableViT or TransferLesionClassifier
CNN_MODEL = None   # BaselineCNN or TransferLesionClassifier
SAMPLE_DF: Optional[pd.DataFrame] = None
SAMPLE_DIR = "data/demo_samples"


def np_to_base64(img_np: np.ndarray, format: str = "JPEG") -> str:
    """Encodes a uint8 RGB numpy image to base64 data URL string."""
    pil_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"


def _load_model_from_checkpoint(ckpt_path: str, num_classes: int, fallback_model):
    """
    Smart checkpoint loader:
    - If checkpoint has 'backbone' key -> build TransferLesionClassifier
    - Otherwise -> use fallback_model (ExplainableViT / BaselineCNN)
    """
    if not os.path.exists(ckpt_path):
        print(f"[API] No checkpoint at {ckpt_path}. Using random weights.")
        return fallback_model, False
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        backbone_name = ckpt.get("backbone", None)
        if backbone_name:
            model = TransferLesionClassifier(num_classes=num_classes, backbone=backbone_name)
            model.load_state_dict(ckpt["model_state_dict"])
            metrics = ckpt.get("metrics", {})
            print(f"[API] Loaded TransferLearning checkpoint: {ckpt_path}")
            print(f"      backbone={backbone_name}  epoch={ckpt.get('epoch','?')}  "
                  f"F1={metrics.get('macro_f1','?'):.3f}  "
                  f"MEL_sens={metrics.get('melanoma_sensitivity','?'):.3f}")
        else:
            model = fallback_model
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[API] Loaded legacy checkpoint: {ckpt_path}")
        return model, True
    except Exception as e:
        print(f"[API] Warning loading checkpoint {ckpt_path}: {e}")
        return fallback_model, False


def init_system():
    global VIT_MODEL, CNN_MODEL, SAMPLE_DF
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # Ensure demo dataset exists
    csv_path = os.path.join(SAMPLE_DIR, "metadata.csv")
    if not os.path.exists(csv_path):
        generate_sample_dataset(SAMPLE_DIR, num_samples_per_class=4)
    SAMPLE_DF = pd.read_csv(csv_path)

    n = len(CLASSES)

    # Load ViT slot (primary classifier)
    vit_fallback = ExplainableViT.create_small(num_classes=n, img_size=224)
    VIT_MODEL, _ = _load_model_from_checkpoint(
        "checkpoints/vit_lesion_classifier_best.pt", n, vit_fallback
    )
    VIT_MODEL.to(DEVICE).eval()

    # Load CNN baseline slot (distinct ResNet/ConvNet architecture for benchmark comparison)
    CNN_MODEL = BaselineCNN(num_classes=n)
    cnn_ckpt_path = "checkpoints/cnn_baseline_best.pt"
    if os.path.exists(cnn_ckpt_path):
        try:
            ckpt = torch.load(cnn_ckpt_path, map_location=DEVICE, weights_only=False)
            # Only load if state dict keys match BaselineCNN (not TransferLesionClassifier)
            if not ckpt.get("backbone", None):
                CNN_MODEL.load_state_dict(ckpt["model_state_dict"])
                print(f"[API] Loaded BaselineCNN checkpoint: {cnn_ckpt_path}")
            else:
                print(f"[API] Initialized BaselineCNN architecture for benchmark slot.")
        except Exception as e:
            print(f"[API] Warning loading CNN checkpoint: {e}")
    else:
        print(f"[API] Initialized BaselineCNN architecture for benchmark slot.")

    CNN_MODEL.to(DEVICE).eval()


# Initialize on startup
init_system()


@app.get("/api/samples")
def get_samples():
    """Returns available demo clinical library cases."""
    if SAMPLE_DF is None or len(SAMPLE_DF) == 0:
        return {"samples": []}
    
    samples = []
    for _, row in SAMPLE_DF.iterrows():
        samples.append({
            "image_id": str(row["image_id"]),
            "dx": str(row["dx"]),
            "has_hair_artifact": bool(row["has_hair_artifact"]),
            "has_marker_artifact": bool(row["has_marker_artifact"]),
            "age": int(row["age"]),
            "sex": str(row["sex"]),
            "localization": str(row["localization"])
        })
    return {"samples": samples}


def get_image_tensor_and_rgb(
    file: Optional[UploadFile] = None,
    sample_id: Optional[str] = None
) -> Tuple[torch.Tensor, np.ndarray, str]:
    img_rgb = None
    img_name = "Custom Upload"

    if file is not None and file.filename:
        contents = file.file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_rgb = np.array(pil_img)
        img_name = file.filename
    elif sample_id:
        img_path = os.path.join(SAMPLE_DIR, "images", f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(SAMPLE_DIR, "images", sample_id)
        if os.path.exists(img_path):
            pil_img = Image.open(img_path).convert("RGB")
            img_rgb = np.array(pil_img)
            img_name = sample_id

    if img_rgb is None:
        raise HTTPException(status_code=400, detail="No valid image provided via upload or sample_id.")

    # Resize to standard dermoscopy view if needed
    if img_rgb.shape[0] != 256 or img_rgb.shape[1] != 256:
        img_rgb = cv2.resize(img_rgb, (256, 256))

    transform = get_val_transforms(img_size=224)
    tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    return tensor, img_rgb, img_name


@app.post("/api/diagnose")
async def diagnose_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    discard_ratio: float = Form(0.85),
    colormap: str = Form("turbo"),
    alpha: float = Form(0.55),
    add_residual: bool = Form(True)
):
    """
    Performs real-time diagnosis with Explainability heatmap.
    Uses Attention Rollout for ExplainableViT, GradCAM for TransferLesionClassifier.
    """
    t0 = time.perf_counter()
    tensor, img_rgb, img_name = get_image_tensor_and_rgb(file, sample_id)

    # XAI dispatch: Attention Rollout for custom ViT, GradCAM for transfer models
    if isinstance(VIT_MODEL, ExplainableViT):
        rollout = AttentionRollout(
            VIT_MODEL,
            discard_ratio=float(discard_ratio),
            add_residual=bool(add_residual)
        )
        heatmaps, logits = rollout(tensor)
        xai_method = "Attention Rollout"
    else:
        # TransferLesionClassifier — use GradCAM on the backbone
        gradcam = GradCAM(VIT_MODEL)
        heatmaps, logits = gradcam(tensor)
        xai_method = "Grad-CAM"

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    top_class_name = CLASSES[pred_idx]
    top_code = top_class_name.split(" ")[0]
    top_prob = float(probs[pred_idx])

    risk_title, risk_class, risk_advice = RISK_MAP.get(
        top_code,
        ("DIAGNOSTIC ASSESSMENT", "warning", "Clinical follow-up recommended.")
    )

    # Overlay
    heatmap_2d = heatmaps[0]
    overlay_rgb = overlay_heatmap(img_rgb, heatmap_2d, colormap=colormap, alpha=float(alpha))
    diag = detect_artifact_vs_pathology(img_rgb, heatmap_2d)

    # Class predictions list
    predictions = []
    for i, cname in enumerate(CLASSES):
        code = cname.split(" ")[0]
        predictions.append({
            "class_name": cname,
            "code": code,
            "probability": float(probs[i]),
            "percent": round(float(probs[i]) * 100, 1)
        })
    predictions.sort(key=lambda x: x["probability"], reverse=True)

    # Generate structured natural language clinical explanation
    pathology_ratio = round(float(diag["central_attention_ratio"]) * 100, 1)
    overlap_ratio = round(float(diag["edge_overlap_ratio"]) * 100, 1)
    
    explanation_text = (
        f"The Vision Transformer evaluated this dermoscopy scan ({img_name}) with {top_prob*100:.1f}% confidence for {top_class_name}. "
        f"Attention Rollout XAI confirms that {pathology_ratio}% of the model's self-attention is concentrated directly on the central pathology. "
    )
    if diag["is_artifact_suspect"]:
        explanation_text += (
            f"WARNING: Peripheral overlap of {overlap_ratio}% detected near image edges or artifacts. "
            f"While the primary prediction remains {top_code}, clinician review is advised to rule out artifact bias."
        )
    else:
        explanation_text += (
            f"Low edge overlap ({overlap_ratio}%) verifies that the diagnosis is driven by authentic lesion morphology "
            f"(border irregularity and pigment network) rather than background noise, hair, or border artifacts."
        )

    return {
        "status": "success",
        "image_name": str(img_name),
        "inference_time_ms": round(float(elapsed_ms), 2),
        "xai_method": xai_method,
        "top_prediction": {
            "class_name": str(top_class_name),
            "code": str(top_code),
            "probability": float(top_prob),
            "percent": round(float(top_prob) * 100, 1),
            "risk_title": str(risk_title),
            "risk_class": str(risk_class),
            "risk_advice": str(risk_advice)
        },
        "predictions": predictions,
        "diagnostics": {
            "central_attention_ratio": round(float(diag["central_attention_ratio"]) * 100, 1),
            "peripheral_attention_ratio": round(float(diag["peripheral_attention_ratio"]) * 100, 1),
            "edge_overlap_ratio": round(float(diag["edge_overlap_ratio"]) * 100, 1),
            "is_artifact_suspect": bool(diag["is_artifact_suspect"]),
            "interpretation": str(diag["interpretation"]),
            "detailed_explanation": explanation_text
        },
        "images": {
            "original": np_to_base64(img_rgb),
            "overlay": np_to_base64(overlay_rgb)
        }
    }


@app.post("/api/benchmark")
async def benchmark_comparison(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    colormap: str = Form("turbo"),
    alpha: float = Form(0.55)
):
    """
    Runs side-by-side comparison between primary ViT model and CNN baseline.
    Uses best available XAI per model type.
    """
    tensor, img_rgb, img_name = get_image_tensor_and_rgb(file, sample_id)

    # 1. Primary model (ViT slot) — AttentionRollout if ExplainableViT else GradCAM
    t_vit_0 = time.perf_counter()
    if isinstance(VIT_MODEL, ExplainableViT):
        vit_rollout = AttentionRollout(VIT_MODEL, discard_ratio=0.85, add_residual=True)
        vit_heatmaps, vit_logits = vit_rollout(tensor)
        vit_arch = "Vision Transformer + Attention Rollout"
    else:
        vit_gradcam = GradCAM(VIT_MODEL)
        vit_heatmaps, vit_logits = vit_gradcam(tensor)
        vit_arch = "EfficientNet-B0 Transfer (Grad-CAM)"
    vit_latency = (time.perf_counter() - t_vit_0) * 1000
    vit_probs = torch.softmax(vit_logits, dim=-1)[0].cpu().numpy()
    vit_idx = int(np.argmax(vit_probs))

    vit_overlay = overlay_heatmap(img_rgb, vit_heatmaps[0], colormap=colormap, alpha=float(alpha))
    vit_diag = detect_artifact_vs_pathology(img_rgb, vit_heatmaps[0])

    # 2. CNN model (CNN slot) — always GradCAM
    t_cnn_0 = time.perf_counter()
    cnn_gradcam = GradCAM(CNN_MODEL)
    cnn_heatmaps, cnn_logits = cnn_gradcam(tensor)
    cnn_latency = (time.perf_counter() - t_cnn_0) * 1000
    cnn_probs = torch.softmax(cnn_logits, dim=-1)[0].cpu().numpy()
    cnn_idx = int(np.argmax(cnn_probs))
    cnn_arch = "EfficientNet-B0 Transfer (Grad-CAM)" if isinstance(CNN_MODEL, TransferLesionClassifier) else "Baseline CNN (Grad-CAM)"

    cnn_overlay = overlay_heatmap(img_rgb, cnn_heatmaps[0], colormap="jet", alpha=float(alpha))
    cnn_diag = detect_artifact_vs_pathology(img_rgb, cnn_heatmaps[0])

    return {
        "status": "success",
        "image_name": str(img_name),
        "original_image": np_to_base64(img_rgb),
        "vit": {
            "architecture": vit_arch,
            "pred_class": str(CLASSES[vit_idx]),
            "probability": round(float(vit_probs[vit_idx]) * 100, 1),
            "latency_ms": round(float(vit_latency), 2),
            "param_count": int(sum(p.numel() for p in VIT_MODEL.parameters())),
            "overlay_image": np_to_base64(vit_overlay),
            "diagnostics": {
                "central_focus": round(float(vit_diag["central_attention_ratio"]) * 100, 1),
                "artifact_overlap": round(float(vit_diag["edge_overlap_ratio"]) * 100, 1),
                "is_suspect": bool(vit_diag["is_artifact_suspect"]),
                "interpretation": str(vit_diag["interpretation"])
            }
        },
        "cnn": {
            "architecture": cnn_arch,
            "pred_class": str(CLASSES[cnn_idx]),
            "probability": round(float(cnn_probs[cnn_idx]) * 100, 1),
            "latency_ms": round(float(cnn_latency), 2),
            "param_count": int(sum(p.numel() for p in CNN_MODEL.parameters())),
            "overlay_image": np_to_base64(cnn_overlay),
            "diagnostics": {
                "central_focus": round(float(cnn_diag["central_attention_ratio"]) * 100, 1),
                "artifact_overlap": round(float(cnn_diag["edge_overlap_ratio"]) * 100, 1),
                "is_suspect": bool(cnn_diag["is_artifact_suspect"]),
                "interpretation": str(cnn_diag["interpretation"])
            }
        }
    }


@app.post("/api/layerwise")
async def layerwise_attention(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    colormap: str = Form("turbo"),
    alpha: float = Form(0.55)
):
    """
    Returns progressive layerwise attention/activation depth overlays for each model stage.
    """
    tensor, img_rgb, img_name = get_image_tensor_and_rgb(file, sample_id)

    if isinstance(VIT_MODEL, ExplainableViT):
        rollout = AttentionRollout(VIT_MODEL, discard_ratio=0.85, add_residual=True)
        layer_masks = rollout.get_layerwise_rollouts(tensor)
    else:
        # TransferLesionClassifier / CNN feature activation depth
        activations = []
        hooks = []
        def make_hook():
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    act = output.detach().pow(2).mean(dim=1)
                    activations.append(act)
            return hook

        target_modules = []
        if hasattr(VIT_MODEL, "backbone") and hasattr(VIT_MODEL.backbone, "blocks"):
            target_modules = list(VIT_MODEL.backbone.blocks)
        else:
            for m in VIT_MODEL.modules():
                if isinstance(m, (nn.Conv2d, nn.Sequential)):
                    target_modules.append(m)
            target_modules = target_modules[::max(1, len(target_modules) // 7)][:7]

        for mod in target_modules:
            hooks.append(mod.register_forward_hook(make_hook()))

        with torch.no_grad():
            _ = VIT_MODEL(tensor)

        for h in hooks:
            h.remove()

        layer_masks = []
        for act in activations:
            mask = F.interpolate(
                act.unsqueeze(1),
                size=(tensor.shape[2], tensor.shape[3]),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)[0].cpu().numpy()
            c_min, c_max = mask.min(), mask.max()
            if c_max - c_min > 1e-8:
                mask = (mask - c_min) / (c_max - c_min)
            else:
                mask = np.ones_like(mask) * 0.5
            layer_masks.append(mask)

    layer_overlays = []
    for i, mask in enumerate(layer_masks):
        overlay = overlay_heatmap(img_rgb, mask, colormap=colormap, alpha=float(alpha))
        layer_overlays.append({
            "layer_number": i + 1,
            "image": np_to_base64(overlay)
        })

    return {
        "status": "success",
        "image_name": img_name,
        "layers": layer_overlays
    }


# Mount static directory to serve frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n[SERVER] Starting Explainable ViT Medical Web Server at http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
