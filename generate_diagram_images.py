import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_architecture_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    # Title
    ax.text(50, 97, "Explainable Vision Transformer (ViT) Diagnostic Framework", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(50, 94, "Complete Multi-Tier System Architecture & Explainability Engine", 
            ha="center", va="center", fontsize=11, color="#475569")

    # Helper function for drawing styled cards
    def draw_box(x, y, w, h, title, bg_color, border_color, text_color="#1E293B"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.2",
                                      linewidth=1.8, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(rect)
        if title:
            ax.text(x + 2, y + h - 2.5, title, fontsize=10, fontweight="bold", color=border_color)

    def draw_inner_card(x, y, w, h, title, subtitle="", bg="#FFFFFF", border="#CBD5E1", title_color="#0F172A"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
                                      linewidth=1.2, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (1.2 if subtitle else 0), title, ha="center", va="center", 
                fontsize=9, fontweight="bold", color=title_color)
        if subtitle:
            ax.text(x + w/2, y + h/2 - 1.5, subtitle, ha="center", va="center", 
                    fontsize=7.5, color="#64748B")

    # 1. Presentation Tier (Top)
    draw_box(4, 76, 92, 15, "1. PRESENTATION LAYER (Clinical Workstation & Web Dashboard)", "#EFF6FF", "#2563EB")
    draw_inner_card(6, 78, 20, 9.5, "Image Ingestion Dropzone", "Upload (JPG/PNG) & Preloaded Library", bg="#FFFFFF", border="#93C5FD")
    draw_inner_card(28.5, 78, 20, 9.5, "Live Attention Rollout Viewer", "Turbo / Jet / Viridis Colormaps & Alpha", bg="#FFFFFF", border="#93C5FD")
    draw_inner_card(51, 78, 20, 9.5, "Malignancy Risk Stratification", "High Risk / Moderate / Benign Warnings", bg="#FFFFFF", border="#93C5FD")
    draw_inner_card(73.5, 78, 20.5, 9.5, "Artifact Interference Bar", "Central Focus % & Hair Bias Alerts", bg="#FFFFFF", border="#93C5FD")

    # 2. API Gateway (FastAPI)
    draw_box(4, 59, 92, 13, "2. API GATEWAY & ASYNCHRONOUS BACKEND LAYER (FastAPI server.py)", "#F1F5F9", "#475569")
    draw_inner_card(6, 61, 20, 8, "POST /api/diagnose", "ViT Inference & Attention Rollout", bg="#FFFFFF", border="#94A3B8")
    draw_inner_card(28.5, 61, 20, 8, "POST /api/benchmark", "ViT vs CNN Grad-CAM Comparison", bg="#FFFFFF", border="#94A3B8")
    draw_inner_card(51, 61, 20, 8, "POST /api/layerwise", "Progressive Attention Depth", bg="#FFFFFF", border="#94A3B8")
    draw_inner_card(73.5, 61, 20.5, 8, "GET /api/samples", "Clinical Metadata Retrieval", bg="#FFFFFF", border="#94A3B8")

    # 3. Core ML Models (ViT & CNN)
    draw_box(4, 28, 44, 27, "3A. VISION TRANSFORMER CORE (Proposed)", "#ECFDF5", "#059669")
    draw_inner_card(6, 44, 40, 7.5, "Patch Embedding & Linear Projection", "196 Patches of 16x16 + [CLS] Token (D=384)", bg="#FFFFFF", border="#6EE7B7")
    draw_inner_card(6, 35.5, 40, 7.5, "12x Transformer Encoder Blocks", "Pre-LN, Multi-Head Self-Attention, GELU MLP", bg="#FFFFFF", border="#6EE7B7")
    draw_inner_card(6, 30, 19, 4.5, "Softmax Head (7 Classes)", "MEL, NV, BCC, AKIEC, etc.", bg="#FFFFFF", border="#6EE7B7")
    draw_inner_card(27, 30, 19, 4.5, "Attention Recorder Hook", "Extracts [B, 6, 197, 197]", bg="#FFFFFF", border="#6EE7B7")

    draw_box(52, 28, 44, 27, "3B. BASELINE CNN (Benchmarking)", "#FEF2F2", "#DC2626")
    draw_inner_card(54, 44, 40, 7.5, "Convolutional Feature Extractor", "Multi-stage Conv & Pooling (Local 3x3 Kernels)", bg="#FFFFFF", border="#FCA5A5")
    draw_inner_card(54, 35.5, 40, 7.5, "Global Avg Pool & Classification", "Linear Head (7 Diagnostic Classes)", bg="#FFFFFF", border="#FCA5A5")
    draw_inner_card(54, 30, 40, 4.5, "Grad-CAM Gradient Hook", "Target Activation & Backward Gradients", bg="#FFFFFF", border="#FCA5A5")

    # 4. Explainable AI & Storage Tier (Bottom)
    draw_box(4, 4, 60, 20, "4. EXPLAINABLE AI (XAI) REASONING ENGINE", "#FAF5FF", "#7C3AED")
    draw_inner_card(6, 13.5, 27, 7.5, "Attention Rollout Recursion", "R = ∏ (0.5·A_l + 0.5·I) Across Layers", bg="#FFFFFF", border="#C084FC")
    draw_inner_card(35, 13.5, 27, 7.5, "[CLS] to Patch Bicubic Upsampling", "Grid (14x14) -> Input (224x224)", bg="#FFFFFF", border="#C084FC")
    draw_inner_card(6, 6, 27, 6.5, "Artifact Susceptibility Analyzer", "Pathology Center vs Peripheral Overlap", bg="#FFFFFF", border="#C084FC")
    draw_inner_card(35, 6, 27, 6.5, "Colormap Overlay Generator", "Turbo / Jet Heatmap Blending", bg="#FFFFFF", border="#C084FC")

    draw_box(68, 4, 28, 20, "5. STORAGE & REPOSITORY", "#FFFBEB", "#D97706")
    draw_inner_card(70, 13.5, 24, 7.5, "ISIC 2024 / HAM10000", "Multi-Class Dermoscopy Images & Metadata", bg="#FFFFFF", border="#FCD34D")
    draw_inner_card(70, 6, 24, 6.5, "Model Checkpoints (.pt)", "ViT-B/16 & CNN Trained Weights", bg="#FFFFFF", border="#FCD34D")

    # Flow Arrows
    def draw_arrow(x1, y1, x2, y2, color="#64748B"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=3, shrinkB=3))

    draw_arrow(50, 76, 50, 72, "#2563EB")
    draw_arrow(26, 59, 26, 55, "#059669")
    draw_arrow(74, 59, 74, 55, "#DC2626")
    draw_arrow(26, 28, 26, 24, "#7C3AED")
    draw_arrow(74, 28, 74, 24, "#D97706")

    plt.tight_layout()
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Architecture Diagram -> {output_path}")


def create_data_flow_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#F8FAFC")

    # Title
    ax.text(50, 97, "Explainable Vision Transformer (ViT) - Data Flow Diagram (DFD)", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#0F172A")
    ax.text(50, 94, "Detailed End-to-End Pipeline: Ingestion, Multi-Head Attention Flow & Diagnostic Output", 
            ha="center", va="center", fontsize=11, color="#475569")

    def draw_step_card(x, y, w, h, step_num, title, items, bg="#FFFFFF", border="#2563EB", tag=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.0",
                                      linewidth=1.6, edgecolor=border, facecolor=bg)
        ax.add_patch(rect)
        # Step header
        ax.text(x + 2, y + h - 2.8, f"STEP {step_num}: {title.upper()}", fontsize=9.5, fontweight="bold", color=border)
        if tag:
            ax.text(x + w - 2, y + h - 2.8, tag, ha="right", fontsize=8, fontweight="bold", color="#64748B")
        
        # Items list
        for i, item in enumerate(items):
            ax.text(x + 3, y + h - 5.5 - (i * 2.3), f"• {item}", fontsize=8.5, color="#334155")

    # Step 1: Input & Ingestion
    draw_step_card(4, 76, 43, 15, "1", "Dermoscopy Ingestion & Augmentation", [
        "High-Resolution Dermoscopy Image (RGB, H x W x 3)",
        "Image Resizing & Tensor Scaling to (3, 224, 224)",
        "Hair Artifact Simulation & Color Jitter Augmentation",
        "ImageNet Channel Normalization (μ=[0.485,...], σ=[0.229,...])"
    ], bg="#EFF6FF", border="#2563EB", tag="Input Ingestion")

    # Step 2: Patch Tokenization
    draw_step_card(53, 76, 43, 15, "2", "Patch Tokenization & Positional Encoding", [
        "Divide image into 196 non-overlapping 16x16 patches",
        "Linear Projection: X_p ∈ ℝ^(196 x 384)",
        "Prepend learnable [CLS] classification token (Z_0 ∈ ℝ^(197 x 384))",
        "Add 1D learnable Positional Embeddings: E_pos ∈ ℝ^(197 x 384)"
    ], bg="#F0FDF4", border="#16A34A", tag="Embedding")

    # Step 3: Multi-Head Self Attention
    draw_step_card(4, 55, 92, 17, "3", "Multi-Head Self-Attention Flow Across 12 Transformer Blocks", [
        "Compute Query, Key, Value Projections: Q = X W_Q,  K = X W_K,  V = X W_V",
        "Scaled Dot-Product Attention: A_l = Softmax(Q_l K_l^T / √d_k) ∈ ℝ^(6 x 197 x 197)",
        "Context Aggregation: Attention_Out = A_l · V, followed by Multi-Layer Perceptron (GELU)",
        "Attention Matrix Retention Hook explicitly caches A_l across all layers l ∈ [1, 12]"
    ], bg="#FAF5FF", border="#9333EA", tag="Transformer Core")

    # Step 4A: Classification Branch
    draw_step_card(4, 32, 43, 19, "4A", "Multi-Class Clinical Risk Head", [
        "Extract final [CLS] token output vector: Z_L[0] ∈ ℝ^384",
        "Linear Classification Layer (384 -> 7 diagnostic classes)",
        "Softmax Probabilities: P(MEL), P(NV), P(BCC), P(AKIEC), etc.",
        "Class-Balanced Focal Loss for heavy imbalance mitigation",
        "Malignancy Risk Stratification: High (MEL/BCC) / Moderate / Benign"
    ], bg="#FEF2F2", border="#DC2626", tag="Classification Branch")

    # Step 4B: Explainability (XAI) Branch
    draw_step_card(53, 32, 43, 19, "4B", "Attention Rollout (XAI) Engine", [
        "Layer-wise Head Fusion (mean / max across 6 heads)",
        "Residual Connection Combination: Â_l = 0.5·A_l + 0.5·I",
        "Recursive Matrix Product: Rollout R = ∏ Â_l",
        "Extract [CLS] -> Patch Attention: R[0, 1:] ∈ ℝ^196",
        "Reshape to 14x14 Grid & Bicubic Upsample to (224, 224)",
        "Min-Max Normalization to [0, 1] range"
    ], bg="#FFFBEB", border="#D97706", tag="Explainability Branch")

    # Step 5: Post-processing & Output
    draw_step_card(4, 7, 92, 21, "5", "Clinical Diagnostic Presentation & Artifact Filtering", [
        "Alpha Heatmap Blending: Overlays Turbo/Jet/Viridis colormap on original dermoscopy image",
        "Artifact vs Pathology Analyzer: Computes Central Lesion Attention % vs Peripheral / Hair Edge Overlap %",
        "Clever Hans Immunity Check: Verifies AI attends to cellular morphology rather than background shortcuts",
        "JSON + Base64 Payload Delivery via FastAPI Asynchronous Endpoints",
        "Real-Time Browser DOM Rendering in Modern Hospital Decision Support Dashboard"
    ], bg="#F8FAFC", border="#0284C7", tag="Presentation & Diagnostics")

    # Arrows
    def draw_arrow(x1, y1, x2, y2, label="", color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.8, label, fontsize=8, fontweight="bold", color=color, ha="center")

    draw_arrow(47, 83.5, 53, 83.5, "Tensor (3, 224, 224)", "#2563EB")
    draw_arrow(74.5, 76, 74.5, 72, "Tokens (197, 384)", "#16A34A")
    draw_arrow(25.5, 55, 25.5, 51, "[CLS] Vector", "#DC2626")
    draw_arrow(74.5, 55, 74.5, 51, "Attention Weights A_l", "#D97706")
    draw_arrow(25.5, 32, 25.5, 28, "Risk Scores", "#DC2626")
    draw_arrow(74.5, 32, 74.5, 28, "Attention Heatmap", "#D97706")

    plt.tight_layout()
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Data Flow Diagram -> {output_path}")


if __name__ == "__main__":
    # Save in docs/ and root
    create_architecture_diagram("docs/architecture_diagram.png")
    create_architecture_diagram("architecture_diagram.png")
    create_data_flow_diagram("docs/data_flow_diagram.png")
    create_data_flow_diagram("data_flow_diagram.png")
    print("All diagram images successfully generated and saved!")
