import torch, os, sys
sys.stdout.reconfigure(line_buffering=True)

for tag, path in [("ViT", "checkpoints/vit_lesion_classifier_best.pt"), ("CNN", "checkpoints/cnn_baseline_best.pt")]:
    if not os.path.exists(path):
        print(f"{tag}: MISSING {path}")
        continue
    size_mb = os.path.getsize(path)/1e6
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck.get("metrics", {})
    pca = m.get("per_class_accuracy", {})
    print(f"\n{'='*55}")
    print(f"  {tag} | {path} ({size_mb:.1f} MB)")
    print(f"  Backbone: {ck.get('backbone','custom')} | Epoch: {ck.get('epoch','?')}")
    print(f"  Accuracy: {m.get('accuracy',0)*100:.1f}%  Macro-F1: {m.get('macro_f1',0):.4f}  MEL-Sens: {m.get('melanoma_sensitivity',0)*100:.1f}%")
    if pca:
        print("  Per-Class:", {k: f"{v*100:.0f}%" for k,v in pca.items()})

print("\nDone.")
