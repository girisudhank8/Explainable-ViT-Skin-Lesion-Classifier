"""Quick diagnostic to check all imports and dataset before full training."""
import sys
import os

print("--- Import check ---", flush=True)

try:
    import pandas as pd
    print("pandas OK", flush=True)
except Exception as e:
    print(f"pandas FAILED: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"torch FAILED: {e}", flush=True)
    sys.exit(1)

try:
    from src.models.vit_classifier import ExplainableViT
    print("ExplainableViT import OK", flush=True)
    model = ExplainableViT.create_small(num_classes=7, img_size=224)
    print(f"ViT created, params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
except Exception as e:
    print(f"ViT FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

try:
    from src.models.cnn_baseline import BaselineCNN
    cnn = BaselineCNN(num_classes=7)
    print(f"CNN created, params: {sum(p.numel() for p in cnn.parameters()):,}", flush=True)
except Exception as e:
    print(f"CNN FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

try:
    from src.data.dataset import SkinLesionDataset
    from src.data.transforms import get_train_transforms, get_val_transforms
    print("Data imports OK", flush=True)
except Exception as e:
    print(f"Data imports FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

try:
    from src.training.trainer import ModelTrainer, TrainingConfig
    print("Trainer import OK", flush=True)
except Exception as e:
    print(f"Trainer FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

try:
    csv_path = "data/demo_samples/metadata.csv"
    print(f"CSV exists: {os.path.exists(csv_path)}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"Dataset rows: {len(df)}", flush=True)
    print(f"Columns: {list(df.columns)}", flush=True)
    counts = df["dx"].value_counts().to_dict()
    print(f"Class distribution: {counts}", flush=True)
    img_dir = "data/demo_samples/images"
    print(f"Image dir exists: {os.path.exists(img_dir)}", flush=True)
    if os.path.exists(img_dir):
        imgs = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        print(f"Image count: {len(imgs)}", flush=True)
except Exception as e:
    print(f"Dataset check FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()

print("--- Diagnosis complete ---", flush=True)
