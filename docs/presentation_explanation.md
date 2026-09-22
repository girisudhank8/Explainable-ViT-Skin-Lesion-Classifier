# Presentation Explanation & Viva Defense Guide
## Explainable Vision Transformers (ViT) for Multi-Class Skin Lesion Classification

---

## Slide 1: Title & Architectural Overview
*Visual: Pipeline ribbon from Dermoscopic Image Input -> Explainable ViT -> Decision Support*

### 1. Core Objective & Key Message
This project directly solves the primary barrier to clinical AI adoption-the **black-box trust deficit** by combining high-accuracy **Vision Transformers (ViTs)** with **layer-wise attention rollout explainability** across the full 7-class diagnostic spectrum of skin lesions.

### 2. Spoken Presentation Script (Verbatim)
> *"Good morning/afternoon, esteemed committee members. Today I am presenting our work on **Explainable Vision Transformers for Multi-Class Skin Lesion Classification**.
>
 > *Dermatology is fundamentally an image-driven medical specialty. However, when deep learning models are introduced into clinical workflows, they are almost universally met with skepticism by clinicians. The reason is simple: conventional models operate as black boxes. A board-certified dermatologist cannot justify performing an invasive surgical excision or altering clinical management based on an arbitrary, unverified probability score.
>
 > *As shown on this overview slide, our system establishes an end-to-end transparent diagnostic loop: it accepts standardized dermoscopic images, processes them through an attention-driven Vision Transformer across the full 7-class ISIC spectrum, and produces both a calibrated risk assessment and a verifiable visual explanation using Attention Rollout and Grad-CAM.*
>
 > *Our design philosophy is not to replace the clinician, but to provide a verifiable second opinion where every output is backed by transparent visual evidence."*

### 3. Technical & Clinical Justifications
* **Why 7 Classes Instead of Binary (Melanoma vs. Benign)?**
  * Binary classification is a severe clinical oversimplification. In practice, a non-melanoma lesion is rarely just "benign"-it could be **Basal Cell Carcinoma (BCC)** requiring Mohs surgery, an **Actinic Keratosis (AKIEC)** requiring topical fluorouracil or cryotherapy, or a harmless **Melanocytic Nevus (NV)** or **Seborrheic Keratosis (BKL)**.
  * Classifying across all 7 ISIC categories (*MEL, NV, BCC, AKIEC, BKL, DF, VASC*) mirrors true clinical pathology protocols.
* **Why Dual Explainability (Attention Rollout + Grad-CAM)?**
  * Most studies provide only Grad-CAM on convolutional networks. By providing **Attention Rollout** for the Transformer and **Grad-CAM** for the CNN baseline, we offer comparative auditability, demonstrating how ViT self-attention captures intricate lesion borders that convolutional pooling blends away.

### 4. Expected Examiner Questions & Model Answers
* **Q: Why choose Vision Transformers over established CNNs like ResNet or EfficientNet for skin lesions?**
  * *Answer:* "CNNs are constrained by local receptive fields that expand only with layer depth, making them struggle with long-range structural dependencies across asymmetric lesion borders. In contrast, Vision Transformers employ Multi-Head Self-Attention from layer one, computing global affinities across all image patches. This directly aligns with dermatological ABCDE criteria-specifically Asymmetry and Border irregularity."

---

## Slide 2: Clinical Motivation & The Diagnostic Gap
*Visual: "The Stakes Are Existential - and the Diagnostic Gap Is Wide"*

### 1. Core Objective & Key Message
Melanoma mortality is dictated by early intervention, yet visual inspection is subjective and standard deep learning models frequently "cheat" by learning clinical artifacts rather than genuine pathology.

### 2. Spoken Presentation Script (Verbatim)
> **"Moving to Slide 2: why are the stakes existential?*
> 
> **When malignant melanoma is identified at Stage I while localized to the epidermis, the 5-year patient survival rate exceeds **99%**. However, if missed and allowed to metastasize (Stage IV), survival plummets to **27%**. Diagnostic timing is literally the difference between life and death.*
> 
> **Currently, visual dermoscopy relies heavily on clinician experience. Published clinical literature documents that inter-observer diagnostic agreement among certified dermatologists ranges between only **60% and 85%**. This leads to high rates of unnecessary biopsies for benign lesions, while early amelanotic or subtle melanomas risk being overlooked.*
> 
> **Furthermore, deep learning models often fail silently due to the *'Clever Hans' effect*. When trained on benchmark datasets, standard CNNs often achieve high accuracy by latching onto confounding artifacts-such as surgical marker ink dark skin hair, gel bubbles, or ruler markings-rather than actual cellular atypia.*
> 
> *Our framework addresses this vulnerability through calibrated probability estimation combined with rigorous artifact-susceptibility verification."*

### 3. Technical & Clinical Justifications
* **Epidemiology & Breslow Thickness:**
  * Tumor invasion depth (Breslow thickness in millimeters) is the single most important histological predictor of survival. Every millimeter of progression significantly worsens prognosis, underscoring the necessity of high-sensitivity screening tools.
* **The 'Clever Hans' Vulnerability in Dermatology:**
  * In clinical datasets, biopsy-confirmed melanoma lesions frequently contain pen markings drawn by surgeons before excision.
  * Standard CNN models can easily associate ink markings with malignancy, producing strong laboratory test metrics that fail catastrophically in real clinical settings. Explainable AI is the primary mathematical safeguard against this failure mode.

### 4. Expected Examiner Questions & Model Answers
* **Q: How does your system detect or prevent 'Clever Hans' bias?**
  * *Answer:* "We engineered an automated **Artifact Susceptibility Analyzer** in the post-inference pipeline. It quantifies the ratio of attention mass concentrated within the central lesion core versus peripheral image regions where hair, rulers, and ink marks typically reside. If attention falls disproportionately on peripheral artifacts, an automated visual alert warns the clinician."

---

## Slide 3: System Architecture
*Visual: 5-Layer End-to-End System Architecture Diagram*

### 1. Core Objective & Key Message
The system is architected as an enterprise-grade, 5-layer clinical microservice that cleanly decouples front-end visualization, RESTful API routing, deep learning inference, XAI computation, and persistence.

### 2. Spoken Presentation Script (Verbatim)
> *"Slide 3 presents our complete **System Architecture**, organized into five decoupled layers:*
> 
> *1. **Presentation Layer**: A modern, responsive clinical web dashboard built with HTML, Tailwind CSS, and vanilla JavaScript. It includes an interactive attention viewer with configurable medical colormaps (Turbo, Jet, Viridis), tri-tier risk stratification badges, and an artifact interference indicator.*
>
 > *2. **API Gateway & Backend Layer**: Powered by **FastAPI**. Unlike monolithic prototype tools such as Streamlit, FastAPI provides high-throughput asynchronous execution, native JSON/Base64 serialization, and sub-second REST endpoints (/api/diagnose, /api/benchmark, /api/layerwise) that are ready for hospital PACS or EMR integration.*
>
 > *3. **Core Model Layer**: Implements our primary **Explainable Vision Transformer** alongside a **Baseline CNN** for live side-by-side benchmarking. The ViT processes 196 image patches through Transformer blocks with integrated attention-recording hooks.*
>
 > *4. **XAI Reasoning Engine**: Computes Attention Rollout by recursively multiplying residual attention matrices across all layers, upsampling the [CLS]-to-patch attention back to the original image dimensions.*
>
 > *5. **Storage & Checkpoint Layer**: Manages calibrated checkpoint weights trained under Focal Loss and verified against class mode collapse."*

### 3. Technical & Clinical Justifications
* **Why FastAPI Over Streamlit?**
  * Streamlit re-executes the complete Python script upon every user interaction, leading to substantial latency, redundant model weight allocations, and poor multi-user scalability.
  * FastAPI operates via an asynchronous event loop (ASGI/Uvicorn), maintaining persistent model instances in memory, guaranteeing sub-second (<150ms) response times, and providing a clean API contract for client applications.
* **Non-Intrusive PyTorch Hooks:**
  * Rather than modifying model architectures, attention weights are captured via PyTorch forward hooks registered on the self-attention modules. This preserves native forward-pass execution speeds without recomputing gradients.

### 4. Expected Examiner Questions & Model Answers
* **Q: How does the benchmarking endpoint compare ViT against CNN?**
  * *Answer:* "The /api/benchmark endpoint executes parallel forward passes through both models on the exact same input tensor. It computes Attention Rollout on the ViT and Grad-CAM on the CNN, measuring inference latency, parameter count, class probabilities, and spatial focus side-by-side so clinicians can observe the differences in spatial reasoning."

---

## Slide 4: Data Flow & Clinical Validation
*Visual: 5-Stage Data Flow Diagram (DFD1) with Mathematical Equations and Pipeline Flow*

### 1. Core Objective & Key Message
The data pipeline transforms raw dermoscopy pixels into verifiable clinical decisions through a mathematically rigorous 5-stage flow featuring patch tokenization, self-attention, Class-Balanced Focal Loss, and recursive Attention Rollout.

### 2. Spoken Presentation Script (Verbatim)
> *"Our final slide depicts the granular **Data Flow Diagram and Validation Pipeline**.
> 
> *In **Stages 1 & 2**, input dermoscopic images are resized to 224x224 with 3 channels and normalized. The spatial image is partitioned into N = 196 non-overlapping 16x16 patches, projected into a 384-dimensional embedding space, prepended with a learnable [CLS] token, and summed with 1D positional encodings.*
> 
> *In **Stage 3**, these tokens pass through 12 Transformer blocks where Multi-Head Self-Attention calculates global affinities. At the output, the data flow splits into two parallel branches:*
> 
> *In **Branch 4A**, the final [CLS] token representation is passed to the linear classifier. Because clinical skin lesion datasets are highly imbalanced, we train using **Class-Balanced Focal Loss** with focusing parameter gamma = 2.0. This prevents gradient dominance by majority benign classes and eliminates mode collapse.*
> 
> *In **Branch 4B**, our Explainable AI engine takes the recorded attention matrices A, incorporates identity residual connections (A_hat = 0.5*A + 0.5*I), and computes the recursive product across all layers. The resulting attention vector is upsampled to 224x224 via bicubic interpolation and normalized to [0, 1].*
> 
> *In **Stage 5**, our validation experiments confirmed **100% Melanoma recall/sensitivity** with high Macro-F1 across all 7 classes, ensuring dependable detection of critical malignancies without generating false-alarm mode collapse on benign nevi."*

### 3. Technical & Clinical Justifications
* **Mathematical Formulation of Attention Rollout (Abnar & Zuidema, 2020):**
  $$\hat{A}_l = 0.5 A_l + 0.5 I$$
  $$R = \prod_{l=1}^{L} \hat{A}_l$$
  * *Why add identity $I$?* Raw attention weights do not account for residual (skip) connections between transformer blocks. Adding $0.5 I$ models information that propagates forward unchanged.
  * *Why recursive multiplication?* Representations at layer $l$ are already linear mixtures of representations from layer $l-1$. Multiplying across layers tracks information routing back to the original input patches.
* **Why Focal Loss Instead of Standard Cross-Entropy?**
  $$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$
  * In standard cross-entropy, abundant easy-to-classify benign nevi (NV) dominate the total gradient, causing the model to under-predict rare classes like Melanoma (MEL) or Dermatofibroma (DF).
  * The modulating factor $(1 - p_t)^\gamma$ automatically suppresses the loss contribution from well-classified samples, forcing model optimization toward difficult and high-risk cases.

### 4. Expected Examiner Questions & Model Answers
* **Q: What were your final validation results on the 7 classes?**
  * *Answer:* "Following calibrated transfer learning, our model achieved **100% Melanoma sensitivity** with zero false-alarm collapse on benign nevi (MEL probability on nevus samples dropped to <1%, while true melanomas scored approx 90%). Overall Macro-F1 reached **0.604**, successfully differentiating high-risk malignancies from benign mimickers."

---

## Quick Reference Summary Table for Defense

| Slide | Core Focus | Key Concepts & Terminology | Key Statistics / Equations |
| :--- | :--- | :--- | :--- |
| **Slide 1** | Project Scope & Vision | 7-Class Spectrum, Auditability, ViT, CDSS | Stage I >99% vs Stage IV 27% survival |
| **Slide 2** | Clinical Motivation | Clever Hans Effect, Diagnostic Concordance | 60% - 85% dermatologist agreement |
| **Slide 3** | System Architecture | FastAPI, Decoupled Microservice, PyTorch Hooks | <200ms inference latency, 5 decoupled tiers |
| **Slide 4** | Data Flow & Mathematical XAI | Tokenization, Focal Loss, Recursive Rollout | \hat{A}_l = 0.5 A_l + 0.5 I, 100% melanoma sensitivity |
