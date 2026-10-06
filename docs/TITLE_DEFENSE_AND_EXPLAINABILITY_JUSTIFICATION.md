# Title Defense & Explainability Justification (XAI)
## Project: Explainable Vision Transformer (ViT) for Multi-Class Skin Lesion Classification

---

### 1. Executive Summary & Core Thesis Statement

> **Question for Defense:** *"Why is this project called 'Explainable Vision Transformer'? How is it different from a standard 'black box' deep learning model?"*

**Core Justification:**
Standard deep neural networks (CNNs or standard ViTs) predict skin lesion diagnoses as black-box probability vectors without disclosing *why* or *where* the network looked to arrive at its decision. 

In clinical dermatology, a high classification accuracy is **insufficient for deployment** because physicians must verify that the model relies on legitimate pathological biomarkers (e.g., asymmetric pigmentation, irregular borders, blue-white veil) rather than spurious artifacts (e.g., surgical gel, hair, ruler markings, or lighting variations).

This project integrates **Explainable AI (XAI)** directly into the architectural pipeline through:
1. **In-Model Attention Mechanism (Vision Transformer):** Utilizing self-attention mechanisms $A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ to calculate explicit token-to-token spatial dependencies across image patches.
2. **Attention Rollout Interpretability:** Implementing recursive matrix multiplication across transformer layers to map information flow from deep representation tokens back to raw input pixel regions.
3. **Comparative XAI Auditing:** Benchmarking Vision Transformer Attention Maps against CNN Grad-CAM saliency maps to prove superior localization and immunity to peripheral background artifacts.
4. **Clinical Decision Support UI:** Translating mathematical explainability maps into readable clinical metrics (Central Pathology Focus vs Peripheral Artifact Overlap) and actionable tri-tier risk recommendations.

---

### 2. Theoretical & Mathematical Foundations of XAI in ViT

#### A. Multi-Head Self-Attention (MHSA) as Native Interpretability
In a Vision Transformer, the input image $I \in \mathbb{R}^{H \times W \times C}$ is split into $N = \frac{HW}{P^2}$ patches. Each patch is projected into an embedding space of dimension $D$.

For each layer $l$, the Attention Matrix $A^{(l)}$ for a single head is computed as:
$$A^{(l)} = \text{Softmax}\left( \frac{Q^{(l)} (K^{(l)})^T}{\sqrt{d_k}} \right)$$

where:
- $Q^{(l)}$ (Query): What feature representation the patch is searching for.
- $K^{(l)}$ (Key): What feature representation each surrounding patch offers.
- $A^{(l)}_{i, j}$: The explicit weight of how much patch $i$ attends to patch $j$.

#### B. Attention Rollout (Abnar & Zuidema)
Standard attention visualization only displays the final layer's attention, which ignores identity skip-connections and intermediate transformations. We implement **Attention Rollout**:

1. **Accounting for Residual Connections:**
   $$\hat{A}^{(l)} = 0.5 A^{(l)} + 0.5 I$$
   *(Equal weighting given to transformed features and identity pass-through).*

2. **Recursive Layer Multiplication:**
   $$R^{(l)} = \hat{A}^{(l)} \times R^{(l-1)}$$
   with $R^{(0)} = I$.

The final matrix $R^{(L)}$ provides a mathematically rigorous, un-biased heatmap of exactly which input patches contributed to the classification token `[CLS]`.

---

### 3. Key Distinctions: ViT XAI vs. CNN Grad-CAM

| Dimension | Baseline CNN (Grad-CAM) | Explainable ViT (Attention Rollout) | Clinical Advantage |
| :--- | :--- | :--- | :--- |
| **Explanation Source** | Gradient backpropagation through final convolutional feature maps. | Direct self-attention matrix interactions across all patch combinations. | ViT offers intrinsic architectural interpretability. |
| **Spatial Resolution** | Low resolution (coarse, blurry heatmaps due to aggressive pooling). | Fine-grained patch-level attention ($16 \times 16$ or $8 \times 8$ resolution). | Precise demarcation of lesion boundaries and micro-features. |
| **Artifact Sensitivity** | Frequently highlights hair, skin folds, and dark corners (Clever Hans effect). | High focus density concentrated on central lesion structures. | Audits and prevents model cheating on spurious correlations. |

---

### 4. Code & Architectural Proof Points

1. **Attention Map Extraction:** `src/xai/attention_rollout.py`
   - `VITAttentionRollout`: Intercepts self-attention weights per layer, computes identity-augmented rollout, and rescales heatmaps to native image resolution.
2. **Grad-CAM Baseline:** `src/xai/gradcam.py`
   - `GradCAM`: Computes target class gradients relative to the final spatial feature map for head-to-head XAI comparisons.
3. **Interactive Visual Diagnostics:** `server.py` & UI
   - Endpoints `/api/diagnose`, `/api/benchmark`, and `/api/layerwise` serve layer-by-layer attention evolutions and comparative saliency overlays in real time.

---

### 5. Summary Answers for Viva / Thesis Defense

* **Q: Why call it Explainable ViT instead of just a classifier?**
  * **A:** Because the system output is a **dual payload**: a diagnostic prediction (e.g., Melanoma 94.2%) paired with an audited, mathematically proven spatial attention footprint that justifies *why* the model arrived at that confidence level.

* **Q: How does this solve real-world medical AI problems?**
  * **A:** It enables clinical trust verification. A doctor can immediately see if a high melanoma score was triggered by abnormal cell boundaries vs an ink mark or hair on the patient's skin.
