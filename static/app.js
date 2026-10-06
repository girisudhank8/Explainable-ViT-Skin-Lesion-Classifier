// Client-Side Logic for Explainable ViT Skin Lesion Decision Support

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentInputMode = "library"; // "library" or "upload"
    let selectedFile = null;
    let availableSamples = [];
    let currentScanContext = "";

    // DOM Elements - Navigation
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // DOM Elements - Source Toggle
    const srcLibBtn = document.getElementById("src-lib-btn");
    const srcUploadBtn = document.getElementById("src-upload-btn");
    const sectionLibrary = document.getElementById("section-library");
    const sectionUpload = document.getElementById("section-upload");
    const sampleSelect = document.getElementById("sample-select");
    const sampleMeta = document.getElementById("sample-meta");
    const metaDx = document.getElementById("meta-dx");
    const metaArtifact = document.getElementById("meta-artifact");
    const metaDetails = document.getElementById("meta-details");

    // DOM Elements - Upload
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const clearFileBtn = document.getElementById("clear-file");

    // DOM Elements - Parameters
    const xaiColormap = document.getElementById("xai-colormap");
    const xaiAlpha = document.getElementById("xai-alpha");
    const alphaVal = document.getElementById("alpha-val");
    const xaiDiscard = document.getElementById("xai-discard");
    const discardVal = document.getElementById("discard-val");
    const xaiResidual = document.getElementById("xai-residual");
    const btnDiagnose = document.getElementById("btn-diagnose");

    // DOM Elements - Results View
    const imgOriginal = document.getElementById("img-original");
    const imgOriginalPh = document.getElementById("img-original-placeholder");
    const imgOverlay = document.getElementById("img-overlay");
    const imgOverlayPh = document.getElementById("img-overlay-placeholder");
    const inferenceTime = document.getElementById("inference-time");

    // DOM Elements - Diagnostics
    const artifactBox = document.getElementById("artifact-status-box");
    const artifactIcon = document.getElementById("artifact-icon");
    const artifactTitle = document.getElementById("artifact-title");
    const artifactText = document.getElementById("artifact-text");
    const valCentral = document.getElementById("val-central");
    const valOverlap = document.getElementById("val-overlap");

    // DOM Elements - Risk Card & Probabilities
    const riskBanner = document.getElementById("risk-banner");
    const riskTitle = document.getElementById("risk-title");
    const topClass = document.getElementById("top-class-name");
    const topProbBadge = document.getElementById("top-prob-badge");
    const riskAdvice = document.getElementById("risk-advice");
    const xaiDetailedExplanation = document.getElementById("xai-detailed-explanation");
    const probList = document.getElementById("prob-list");

    // DOM Elements - Benchmark
    const btnRunBenchmark = document.getElementById("btn-run-benchmark");
    const bmImgOriginal = document.getElementById("bm-img-original");
    const bmImgVit = document.getElementById("bm-img-vit");
    const bmImgCnn = document.getElementById("bm-img-cnn");
    const bmPh1 = document.getElementById("bm-ph-1");
    const bmPh2 = document.getElementById("bm-ph-2");
    const bmPh3 = document.getElementById("bm-ph-3");
    const bmVitMeta = document.getElementById("bm-vit-meta");
    const bmVitPred = document.getElementById("bm-vit-pred");
    const bmVitFocus = document.getElementById("bm-vit-focus");
    const bmCnnMeta = document.getElementById("bm-cnn-meta");
    const bmCnnPred = document.getElementById("bm-cnn-pred");
    const bmCnnFocus = document.getElementById("bm-cnn-focus");

    // DOM Elements - Layerwise
    const btnRunLayerwise = document.getElementById("btn-run-layerwise");
    const layerwiseGrid = document.getElementById("layerwise-grid");

    // --- 1. TAB NAVIGATION ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.dataset.tab;
            tabBtns.forEach(b => {
                b.classList.remove("active");
                b.classList.add("text-slate-500");
            });
            btn.classList.add("active");
            btn.classList.remove("text-slate-500");

            const allContents = document.querySelectorAll(".tab-content");
            allContents.forEach(content => {
                if (content.id === targetTab) {
                    content.classList.remove("hidden");
                } else {
                    content.classList.add("hidden");
                }
            });
        });
    });

    // --- 2. SOURCE TOGGLE ---
    srcLibBtn.addEventListener("click", () => {
        currentInputMode = "library";
        srcLibBtn.classList.add("bg-white", "shadow-sm", "text-blue-600");
        srcLibBtn.classList.remove("text-slate-600");
        srcUploadBtn.classList.remove("bg-white", "shadow-sm", "text-blue-600");
        srcUploadBtn.classList.add("text-slate-600");

        sectionLibrary.classList.remove("hidden");
        sectionUpload.classList.add("hidden");
    });

    srcUploadBtn.addEventListener("click", () => {
        currentInputMode = "upload";
        srcUploadBtn.classList.add("bg-white", "shadow-sm", "text-blue-600");
        srcUploadBtn.classList.remove("text-slate-600");
        srcLibBtn.classList.remove("bg-white", "shadow-sm", "text-blue-600");
        srcLibBtn.classList.add("text-slate-600");

        sectionUpload.classList.remove("hidden");
        sectionLibrary.classList.add("hidden");
    });

    // --- 3. PARAMETER SLIDERS ---
    xaiAlpha.addEventListener("input", (e) => {
        alphaVal.textContent = `${Math.round(e.target.value * 100)}%`;
    });

    xaiDiscard.addEventListener("input", (e) => {
        discardVal.textContent = `${Math.round(e.target.value * 100)}%`;
    });

    // --- 4. DROPZONE HANDLING ---
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });

    function handleFileSelected(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.classList.remove("hidden");
        dropzone.classList.add("hidden");

        // Preview original image immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imgOriginal.src = e.target.result;
            imgOriginal.classList.remove("hidden");
            imgOriginalPh.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    }

    clearFileBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        dropzone.classList.remove("hidden");
        imgOriginal.classList.add("hidden");
        imgOriginalPh.classList.remove("hidden");
    });

    // --- 5. LOAD SAMPLE CASES ---
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            availableSamples = data.samples || [];

            sampleSelect.innerHTML = "";
            if (availableSamples.length === 0) {
                sampleSelect.innerHTML = '<option value="">No sample cases found</option>';
                return;
            }

            availableSamples.forEach((s, idx) => {
                const opt = document.createElement("option");
                opt.value = s.image_id;
                const artifactLabel = s.has_hair_artifact ? " (Hair)" : (s.has_marker_artifact ? " (Marker)" : " (Clean)");
                opt.textContent = `${s.image_id} - ${s.dx}${artifactLabel}`;
                sampleSelect.appendChild(opt);
            });

            // Trigger preview for first item
            onSampleSelected();
        } catch (err) {
            console.error("Failed to load samples:", err);
        }
    }

    sampleSelect.addEventListener("change", onSampleSelected);

    function onSampleSelected() {
        const sampleId = sampleSelect.value;
        const sample = availableSamples.find(s => s.image_id === sampleId);
        if (sample) {
            sampleMeta.classList.remove("hidden");
            metaDx.textContent = `Ground Truth: ${sample.dx}`;
            const artifactStr = sample.has_hair_artifact ? "Hair Artifact Present" : (sample.has_marker_artifact ? "Marker Pen Present" : "Clean Lesion");
            metaArtifact.textContent = artifactStr;
            metaDetails.textContent = `Age: ${sample.age} | Sex: ${sample.sex} | Location: ${sample.localization}`;
        }
    }

    loadSamples();

    // Helper functions for simultaneous execution
    async function runBenchmark() {
        btnRunBenchmark.disabled = true;
        btnRunBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Benchmarking...';
        const formData = new FormData();
        formData.append("colormap", xaiColormap.value);
        formData.append("alpha", xaiAlpha.value);
        if (currentInputMode === "upload") {
            if (selectedFile) formData.append("file", selectedFile);
        } else {
            formData.append("sample_id", sampleSelect.value);
        }
        try {
            const res = await fetch("/api/benchmark", { method: "POST", body: formData });
            if (!res.ok) return;
            const data = await res.json();
            bmImgOriginal.src = data.original_image;
            bmImgOriginal.classList.remove("hidden");
            bmPh1.classList.add("hidden");
            bmImgVit.src = data.vit.overlay_image;
            bmImgVit.classList.remove("hidden");
            bmPh2.classList.add("hidden");
            bmImgCnn.src = data.cnn.overlay_image;
            bmImgCnn.classList.remove("hidden");
            bmPh3.classList.add("hidden");
            bmVitMeta.classList.remove("hidden");
            bmVitPred.textContent = `${data.vit.pred_class} (${data.vit.probability}%)`;
            bmVitFocus.textContent = `${data.vit.diagnostics.central_focus}% Central (${data.vit.latency_ms}ms)`;
            bmCnnMeta.classList.remove("hidden");
            bmCnnPred.textContent = `${data.cnn.pred_class} (${data.cnn.probability}%)`;
            bmCnnFocus.textContent = `${data.cnn.diagnostics.central_focus}% Central (${data.cnn.latency_ms}ms)`;
        } catch (err) {
            console.error(err);
        } finally {
            btnRunBenchmark.disabled = false;
            btnRunBenchmark.textContent = "Run Benchmark Comparison";
        }
    }

    async function runLayerwise() {
        btnRunLayerwise.disabled = true;
        btnRunLayerwise.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Computing Layers...';
        const formData = new FormData();
        formData.append("colormap", xaiColormap.value);
        formData.append("alpha", xaiAlpha.value);
        if (currentInputMode === "upload") {
            if (selectedFile) formData.append("file", selectedFile);
        } else {
            formData.append("sample_id", sampleSelect.value);
        }
        try {
            const res = await fetch("/api/layerwise", { method: "POST", body: formData });
            if (!res.ok) return;
            const data = await res.json();
            layerwiseGrid.innerHTML = "";
            data.layers.forEach(layer => {
                const card = document.createElement("div");
                card.className = "bg-slate-50 p-2 rounded-xl border border-slate-200 text-center space-y-1.5";
                card.innerHTML = `
                    <span class="text-xs font-bold text-slate-700 block">Encoder Layer ${layer.layer_number}</span>
                    <div class="aspect-square bg-slate-200 rounded-lg overflow-hidden">
                        <img src="${layer.image}" class="w-full h-full object-cover">
                    </div>
                `;
                layerwiseGrid.appendChild(card);
            });
        } catch (err) {
            console.error(err);
        } finally {
            btnRunLayerwise.disabled = false;
            btnRunLayerwise.textContent = "Compute Depth Breakdown";
        }
    }

    // --- 6. RUN DIAGNOSIS (ViT + ATTENTION ROLLOUT + SIMULTANEOUS BENCHMARK/LAYERS) ---
    const scanProgContainer = document.getElementById("scan-progress-container");
    const scanProgBar = document.getElementById("scan-progress-bar");
    const scanProgStep = document.getElementById("scan-progress-step");
    const scanProgPercent = document.getElementById("scan-progress-percent");

    btnDiagnose.addEventListener("click", async () => {
        btnDiagnose.disabled = true;
        btnDiagnose.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing ViT Attention Flow...';

        const formData = new FormData();
        formData.append("discard_ratio", xaiDiscard.value);
        formData.append("colormap", xaiColormap.value);
        formData.append("alpha", xaiAlpha.value);
        formData.append("add_residual", xaiResidual.checked);

        if (currentInputMode === "upload") {
            if (!selectedFile) {
                alert("Please select or upload an image first.");
                btnDiagnose.disabled = false;
                btnDiagnose.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run ViT Diagnosis & Attention Rollout';
                return;
            }
            formData.append("file", selectedFile);
        } else {
            formData.append("sample_id", sampleSelect.value);
        }

        // Initialize and animate progress bar
        if (scanProgContainer) {
            scanProgContainer.classList.remove("hidden");
            scanProgBar.style.width = "15%";
            scanProgPercent.textContent = "15%";
            scanProgStep.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-indigo-600"></i> Tokenizing 16×16 Image Patches...';
        }

        const stepTimer1 = setTimeout(() => {
            if (scanProgBar) {
                scanProgBar.style.width = "45%";
                scanProgPercent.textContent = "45%";
                scanProgStep.innerHTML = '<i class="fa-solid fa-brain text-indigo-600"></i> Computing Multi-Head Self-Attention Matrices...';
            }
        }, 300);

        const stepTimer2 = setTimeout(() => {
            if (scanProgBar) {
                scanProgBar.style.width = "75%";
                scanProgPercent.textContent = "75%";
                scanProgStep.innerHTML = '<i class="fa-solid fa-microscope text-indigo-600"></i> Synthesizing Attention Rollout & Consulting Gemini AI...';
            }
        }, 700);

        try {
            // Run Diagnosis, Benchmark, and Layer-Wise Depth simultaneously via Promise.all
            const [diagRes] = await Promise.all([
                fetch("/api/diagnose", { method: "POST", body: formData }),
                runBenchmark(),
                runLayerwise()
            ]);

            clearTimeout(stepTimer1);
            clearTimeout(stepTimer2);

            if (!diagRes.ok) {
                const errData = await diagRes.json();
                throw new Error(errData.detail || "Diagnosis failed");
            }

            if (scanProgBar) {
                scanProgBar.style.width = "100%";
                scanProgPercent.textContent = "100%";
                scanProgStep.innerHTML = '<i class="fa-solid fa-circle-check text-emerald-600"></i> Diagnostic Synthesis & XAI Rollout Completed!';
            }

            const data = await diagRes.json();
            renderDiagnosisResults(data);

            setTimeout(() => {
                if (scanProgContainer) scanProgContainer.classList.add("hidden");
            }, 1800);

        } catch (err) {
            clearTimeout(stepTimer1);
            clearTimeout(stepTimer2);
            if (scanProgContainer) scanProgContainer.classList.add("hidden");
            alert(`Error running diagnosis: ${err.message}`);
        } finally {
            btnDiagnose.disabled = false;
            btnDiagnose.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run ViT Diagnosis & Attention Rollout';
        }
    });

    function renderDiagnosisResults(data) {
        // Latency
        inferenceTime.textContent = `Latency: ${data.inference_time_ms} ms`;

        // Images
        imgOriginal.src = data.images.original;
        imgOriginal.classList.remove("hidden");
        imgOriginalPh.classList.add("hidden");

        imgOverlay.src = data.images.overlay;
        imgOverlay.classList.remove("hidden");
        imgOverlayPh.classList.add("hidden");

        // Risk Card
        const top = data.top_prediction;
        const diag = data.diagnostics;

        topClass.textContent = top.class_name;
        topProbBadge.textContent = `${top.percent}%`;
        riskTitle.textContent = top.risk_title;
        riskAdvice.textContent = top.risk_advice;
        const xaiBox = document.getElementById("xai-detailed-explanation");
        if (xaiBox && diag && diag.detailed_explanation) {
            xaiBox.textContent = diag.detailed_explanation;
        }

        // Populate Deep Audit Modal fields
        const auditFocus = document.getElementById("audit-central-focus");
        const auditOverlap = document.getElementById("audit-edge-overlap");
        const auditGemini = document.getElementById("audit-full-gemini-text");
        const auditAdvice = document.getElementById("audit-advice-text");
        if (auditFocus) auditFocus.textContent = `${diag.central_attention_ratio}%`;
        if (auditOverlap) auditOverlap.textContent = `${diag.edge_overlap_ratio}%`;
        if (auditGemini && diag.detailed_explanation) auditGemini.textContent = diag.detailed_explanation;
        if (auditAdvice) auditAdvice.textContent = top.risk_advice;

        // Populate ABCDE Criteria Tab
        if (data.abcde) {
            const a = data.abcde;
            const elAv = document.getElementById("abcde-a-val");
            const elAl = document.getElementById("abcde-a-lvl");
            const elBv = document.getElementById("abcde-b-val");
            const elBl = document.getElementById("abcde-b-lvl");
            const elCv = document.getElementById("abcde-c-val");
            const elCl = document.getElementById("abcde-c-lvl");
            const elDv = document.getElementById("abcde-d-val");
            const elDl = document.getElementById("abcde-d-lvl");
            const elTv = document.getElementById("abcde-tds-val");
            const elTl = document.getElementById("abcde-tds-lvl");
            const elVerdict = document.getElementById("abcde-verdict-text");

            if (elAv) elAv.textContent = a.asymmetry_score;
            if (elAl) elAl.textContent = a.asymmetry_level;
            if (elBv) elBv.textContent = a.border_score;
            if (elBl) elBl.textContent = a.border_level;
            if (elCv) elCv.textContent = `${a.color_score} σ`;
            if (elCl) elCl.textContent = a.color_level;
            if (elDv) elDv.textContent = `${a.diameter_mm} mm`;
            if (elDl) elDl.textContent = a.diameter_level;
            if (elTv) elTv.textContent = a.total_tds_score;
            if (elTl) elTl.textContent = a.tds_risk;

            if (elVerdict) {
                elVerdict.textContent = (
                    `Comparative AI Evaluation: The Vision Transformer evaluated this lesion with Asymmetry score of ${a.asymmetry_score} (${a.asymmetry_level}) ` +
                    `and Border compactness index of ${a.border_score}. Unlike black-box neural networks that predict uninterpretable class vectors, ` +
                    `the Explainable Vision Transformer grounds its decision on verified morphological criteria (${a.tds_risk}), providing dermatologists ` +
                    `with accountable, clinically verifiable safety evidence.`
                );
            }
        }

        // Update Chatbot context indicator
        currentScanContext = `Image: ${data.image_name} | Top Diagnosis: ${top.class_name} (${top.percent}%) | Risk Level: ${top.risk_title} | Central Attention Focus: ${diag.central_attention_ratio}% | Edge/Hair Overlap: ${diag.edge_overlap_ratio}% | XAI Explanation: ${diag.detailed_explanation}`;
        if (chatContextText) {
            chatContextText.textContent = `Scan Context: ${top.code} (${top.percent}%) | Focus: ${diag.central_attention_ratio}%`;
        }

        // Banner Risk Color
        riskBanner.className = "p-3.5 rounded-xl border space-y-1 mb-4 ";
        if (top.risk_class === "danger") {
            riskBanner.classList.add("bg-red-50", "border-red-200", "text-red-900", "pulse-danger");
            topProbBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-red-600 text-white";
        } else if (top.risk_class === "warning") {
            riskBanner.classList.add("bg-amber-50", "border-amber-200", "text-amber-900");
            topProbBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-amber-500 text-white";
        } else {
            riskBanner.classList.add("bg-emerald-50", "border-emerald-200", "text-emerald-900");
            topProbBadge.className = "text-xs font-bold px-2 py-0.5 rounded-full bg-emerald-600 text-white";
        }

        // Artifact Status Bar
        artifactBox.classList.remove("hidden");
        valCentral.textContent = `${diag.central_attention_ratio}%`;
        valOverlap.textContent = `${diag.edge_overlap_ratio}%`;
        artifactText.textContent = diag.interpretation;

        if (diag.is_artifact_suspect) {
            artifactBox.className = "mt-4 p-3 rounded-lg border text-xs flex items-center justify-between bg-amber-50 border-amber-200 text-amber-900";
            artifactIcon.className = "fa-solid fa-triangle-exclamation text-amber-600 text-base";
            artifactTitle.textContent = "Artifact Warning:";
        } else {
            artifactBox.className = "mt-4 p-3 rounded-lg border text-xs flex items-center justify-between bg-emerald-50 border-emerald-200 text-emerald-900";
            artifactIcon.className = "fa-solid fa-circle-check text-emerald-600 text-base";
            artifactTitle.textContent = "Pathology Focus Verified:";
        }

        // Probability Bars
        probList.innerHTML = "";
        data.predictions.forEach(p => {
            const item = document.createElement("div");
            item.className = "space-y-1";

            const isTop = p.code === top.code;
            const barColor = (p.code === "MEL" || p.code === "BCC") ? "bg-red-500" : (p.code === "AKIEC" ? "bg-amber-500" : "bg-blue-600");

            item.innerHTML = `
                <div class="flex justify-between text-xs ${isTop ? 'font-bold text-slate-900' : 'text-slate-600'}">
                    <span>${p.class_name}</span>
                    <span>${p.percent}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill ${barColor}" style="width: ${p.percent}%"></div>
                </div>
            `;
            probList.appendChild(item);
        });
    }

    // --- 7. BENCHMARK COMPARISON ---
    btnRunBenchmark.addEventListener("click", async () => {
        btnRunBenchmark.disabled = true;
        btnRunBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Comparing...';

        const formData = new FormData();
        formData.append("colormap", xaiColormap.value);
        formData.append("alpha", xaiAlpha.value);

        if (currentInputMode === "upload") {
            if (!selectedFile) {
                alert("Please select or upload an image first.");
                btnRunBenchmark.disabled = false;
                btnRunBenchmark.textContent = "Run Benchmark Comparison";
                return;
            }
            formData.append("file", selectedFile);
        } else {
            formData.append("sample_id", sampleSelect.value);
        }

        try {
            const res = await fetch("/api/benchmark", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Benchmark failed");

            const data = await res.json();
            
            // Render images
            bmImgOriginal.src = data.original_image;
            bmImgOriginal.classList.remove("hidden");
            bmPh1.classList.add("hidden");

            bmImgVit.src = data.vit.overlay_image;
            bmImgVit.classList.remove("hidden");
            bmPh2.classList.add("hidden");

            bmImgCnn.src = data.cnn.overlay_image;
            bmImgCnn.classList.remove("hidden");
            bmPh3.classList.add("hidden");

            // Meta
            bmVitMeta.classList.remove("hidden");
            bmVitPred.textContent = `${data.vit.pred_class} (${data.vit.probability}%)`;
            bmVitFocus.textContent = `${data.vit.diagnostics.central_focus}% Central (${data.vit.latency_ms}ms)`;

            bmCnnMeta.classList.remove("hidden");
            bmCnnPred.textContent = `${data.cnn.pred_class} (${data.cnn.probability}%)`;
            bmCnnFocus.textContent = `${data.cnn.diagnostics.central_focus}% Central (${data.cnn.latency_ms}ms)`;

        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            btnRunBenchmark.disabled = false;
            btnRunBenchmark.textContent = "Run Benchmark Comparison";
        }
    });

    // --- 8. LAYERWISE ATTENTION DEPTH ---
    btnRunLayerwise.addEventListener("click", async () => {
        btnRunLayerwise.disabled = true;
        btnRunLayerwise.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Computing Layers...';

        const formData = new FormData();
        formData.append("colormap", xaiColormap.value);
        formData.append("alpha", xaiAlpha.value);

        if (currentInputMode === "upload") {
            if (!selectedFile) {
                alert("Please select or upload an image first.");
                btnRunLayerwise.disabled = false;
                btnRunLayerwise.textContent = "Compute Depth Breakdown";
                return;
            }
            formData.append("file", selectedFile);
        } else {
            formData.append("sample_id", sampleSelect.value);
        }

        try {
            const res = await fetch("/api/layerwise", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Layerwise computation failed");

            const data = await res.json();
            layerwiseGrid.innerHTML = "";

            data.layers.forEach(layer => {
                const card = document.createElement("div");
                card.className = "bg-slate-50 p-2 rounded-xl border border-slate-200 text-center space-y-1.5";
                card.innerHTML = `
                    <span class="text-xs font-bold text-slate-700 block">Encoder Layer ${layer.layer_number}</span>
                    <div class="aspect-square bg-slate-200 rounded-lg overflow-hidden">
                        <img src="${layer.image}" class="w-full h-full object-cover">
                    </div>
                `;
                layerwiseGrid.appendChild(card);
            });

        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            btnRunLayerwise.disabled = false;
            btnRunLayerwise.textContent = "Compute Depth Breakdown";
        }
    });

    // --- 9. GEMINI CHATBOT WIDGET LOGIC ---
    const chatbotToggleBtn = document.getElementById("chatbot-toggle-btn");
    const chatbotCloseBtn = document.getElementById("chatbot-close-btn");
    const chatbotWindow = document.getElementById("chatbot-window");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");
    const chatContextText = document.getElementById("chat-context-text");

    chatbotToggleBtn.addEventListener("click", () => {
        chatbotWindow.classList.toggle("hidden");
        chatbotToggleBtn.classList.add("hidden");
    });

    chatbotCloseBtn.addEventListener("click", () => {
        chatbotWindow.classList.add("hidden");
        chatbotToggleBtn.classList.remove("hidden");
    });

    const headerChatBtn = document.getElementById("header-chat-btn");
    if (headerChatBtn) {
        headerChatBtn.addEventListener("click", () => {
            chatbotWindow.classList.remove("hidden");
            chatbotToggleBtn.classList.add("hidden");
            chatInput.focus();
        });
    }

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;

        // Append User Message
        appendChatMessage("user", msg);
        chatInput.value = "";

        // Append Loading Message
        const loadingId = appendChatMessage("bot", '<i class="fa-solid fa-spinner fa-spin"></i> Consulting Gemini AI...');

        try {
            const formData = new FormData();
            formData.append("user_message", msg);
            if (currentScanContext) {
                formData.append("context", currentScanContext);
            }

            const res = await fetch("/api/chat", {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            const botMsgEl = document.getElementById(loadingId);
            if (botMsgEl) {
                botMsgEl.innerHTML = `
                    <div class="font-bold text-indigo-600 text-[11px] flex items-center justify-between">
                        <span><i class="fa-solid fa-robot"></i> Gemini Assistant</span>
                        <span class="text-[9px] text-slate-400 font-normal">${data.provider}</span>
                    </div>
                    <p class="text-slate-700 leading-relaxed text-[11px] whitespace-pre-wrap mt-1">${escapeHtml(data.reply)}</p>
                `;
            }
        } catch (err) {
            const botMsgEl = document.getElementById(loadingId);
            if (botMsgEl) {
                botMsgEl.innerHTML = `<span class="text-red-500 font-semibold text-[11px]">Error connecting to AI server.</span>`;
            }
        }
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });

    function appendChatMessage(sender, htmlContent) {
        const msgId = "msg-" + Date.now();
        const msgDiv = document.createElement("div");
        if (sender === "user") {
            msgDiv.className = "bg-indigo-600 text-white p-3 rounded-xl ml-6 text-right shadow-xs";
            msgDiv.innerHTML = `<p class="text-[11px] leading-relaxed">${escapeHtml(htmlContent)}</p>`;
        } else {
            msgDiv.className = "bg-white p-3 rounded-xl border border-slate-200 shadow-xs space-y-1";
            msgDiv.id = msgId;
            msgDiv.innerHTML = htmlContent;
        }
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgId;
    }

    function escapeHtml(text) {
        return text.replace(/[&<"']/g, function(m) {
            return { '&': '&amp;', '<': '&lt;', '"': '&quot;', "'": '&#039;' }[m];
        });
    }

    // --- 10. DEEP CLINICAL XAI AUDIT MODAL LOGIC ---
    const btnOpenDeepAudit = document.getElementById("btn-open-deep-audit");
    const deepAuditModal = document.getElementById("deep-audit-modal");
    const btnCloseDeepAudit = document.getElementById("btn-close-deep-audit");
    const btnModalDismiss = document.getElementById("btn-modal-dismiss");

    if (btnOpenDeepAudit && deepAuditModal) {
        btnOpenDeepAudit.addEventListener("click", () => {
            deepAuditModal.classList.remove("hidden");
        });
    }

    if (btnCloseDeepAudit && deepAuditModal) {
        btnCloseDeepAudit.addEventListener("click", () => {
            deepAuditModal.classList.add("hidden");
        });
    }

    if (btnModalDismiss && deepAuditModal) {
        btnModalDismiss.addEventListener("click", () => {
            deepAuditModal.classList.add("hidden");
        });
    }

    // Close modal on escape key
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && deepAuditModal && !deepAuditModal.classList.contains("hidden")) {
            deepAuditModal.classList.add("hidden");
        }
    });

    // --- 11. COUNTERFACTUAL WHAT-IF REASONING ---
    const btnRunCounterfactual = document.getElementById("btn-run-counterfactual");
    const cfImgOrig = document.getElementById("cf-img-orig");
    const cfImgPert = document.getElementById("cf-img-pert");
    const cfPh1 = document.getElementById("cf-ph-1");
    const cfPh2 = document.getElementById("cf-ph-2");
    const cfMetaOrig = document.getElementById("cf-meta-orig");
    const cfMetaPert = document.getElementById("cf-meta-pert");
    const cfInterpBox = document.getElementById("cf-interpretation-box");

    if (btnRunCounterfactual) {
        btnRunCounterfactual.addEventListener("click", async () => {
            btnRunCounterfactual.disabled = true;
            btnRunCounterfactual.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Perturbation...';
            const formData = new FormData();
            if (currentInputMode === "upload") {
                if (selectedFile) formData.append("file", selectedFile);
            } else {
                formData.append("sample_id", sampleSelect.value);
            }

            try {
                const res = await fetch("/api/counterfactual", { method: "POST", body: formData });
                if (!res.ok) throw new Error("Counterfactual analysis failed");
                const data = await res.json();
                
                cfImgOrig.src = data.original_image;
                cfImgOrig.classList.remove("hidden");
                cfPh1.classList.add("hidden");
                cfMetaOrig.textContent = `Baseline: ${data.original_prediction.class} (${data.original_prediction.probability}%)`;
                cfMetaOrig.classList.remove("hidden");

                cfImgPert.src = data.perturbed_image;
                cfImgPert.classList.remove("hidden");
                cfPh2.classList.add("hidden");
                cfMetaPert.textContent = `Post-Suppression: ${data.counterfactual_prediction.class} (${data.counterfactual_prediction.probability}%)`;
                cfMetaPert.classList.remove("hidden");

                cfInterpBox.textContent = data.interpretation;
                cfInterpBox.classList.remove("hidden");
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                btnRunCounterfactual.disabled = false;
                btnRunCounterfactual.textContent = "Run Counterfactual Test";
            }
        });
    }

    // --- 12. CLEVER HANS ARTIFACT DEFENSE (DULLRAZOR) ---
    const btnRunDullrazor = document.getElementById("btn-run-dullrazor");
    const drImgRaw = document.getElementById("dr-img-raw");
    const drImgClean = document.getElementById("dr-img-clean");
    const drImgOverlay = document.getElementById("dr-img-overlay");
    const drPh1 = document.getElementById("dr-ph-1");
    const drPh2 = document.getElementById("dr-ph-2");
    const drPh3 = document.getElementById("dr-ph-3");
    const drStatsBox = document.getElementById("dr-stats-box");

    if (btnRunDullrazor) {
        btnRunDullrazor.addEventListener("click", async () => {
            btnRunDullrazor.disabled = true;
            btnRunDullrazor.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Inpainting Artifacts...';
            const formData = new FormData();
            if (currentInputMode === "upload") {
                if (selectedFile) formData.append("file", selectedFile);
            } else {
                formData.append("sample_id", sampleSelect.value);
            }

            try {
                const res = await fetch("/api/neutralize-artifacts", { method: "POST", body: formData });
                if (!res.ok) throw new Error("Artifact neutralization failed");
                const data = await res.json();

                drImgRaw.src = data.raw_image;
                drImgRaw.classList.remove("hidden");
                drPh1.classList.add("hidden");

                drImgClean.src = data.cleaned_image;
                drImgClean.classList.remove("hidden");
                drPh2.classList.add("hidden");

                drImgOverlay.src = data.cleaned_overlay;
                drImgOverlay.classList.remove("hidden");
                drPh3.classList.add("hidden");

                drStatsBox.innerHTML = `
                    <span>Post-Cleaning Diagnosis: <strong>${data.top_class} (${data.confidence}%)</strong></span>
                    <span>Pathology Focus: <strong>${data.central_focus}% Central</strong> (Noise: ${data.edge_overlap}%)</span>
                `;
                drStatsBox.classList.remove("hidden");
            } catch (err) {
                alert(`Error: ${err.message}`);
            } finally {
                btnRunDullrazor.disabled = false;
                btnRunDullrazor.textContent = "Neutralize Artifacts";
            }
        });
    }

    // --- 13. CERTIFIED PDF REPORT DOWNLOAD ---
    const btnDownloadPdf = document.getElementById("btn-download-pdf");
    if (btnDownloadPdf) {
        btnDownloadPdf.addEventListener("click", async () => {
            btnDownloadPdf.disabled = true;
            btnDownloadPdf.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF Report...';
            const formData = new FormData();
            if (currentInputMode === "upload") {
                if (selectedFile) formData.append("file", selectedFile);
            } else {
                formData.append("sample_id", sampleSelect.value);
            }

            try {
                const res = await fetch("/api/download-report", { method: "POST", body: formData });
                if (!res.ok) throw new Error("PDF generation failed");
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.style.display = "none";
                a.href = url;
                a.download = `MedVision_Report_${Date.now()}.pdf`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            } catch (err) {
                alert(`Error generating PDF: ${err.message}`);
            } finally {
                btnDownloadPdf.disabled = false;
                btnDownloadPdf.innerHTML = '<i class="fa-solid fa-file-pdf text-red-400"></i> Download Certified Clinical Report (PDF)';
            }
        });
    }

    // --- 14. CLINICIAN HUMAN-IN-THE-LOOP (HITL) FEEDBACK ---
    const hitlForm = document.getElementById("hitl-feedback-form");
    const hitlSuccessMsg = document.getElementById("hitl-success-msg");

    if (hitlForm) {
        hitlForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btnSubmit = document.getElementById("btn-submit-hitl");
            btnSubmit.disabled = true;
            btnSubmit.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Logging Audit...';

            const activeImgName = fileName ? fileName.textContent : (sampleSelect.value || "Unknown_Scan");
            const activeTopClass = topClass ? topClass.textContent : "Unknown";
            const rating = document.getElementById("hitl-attention-rating").value;
            const agreement = document.getElementById("hitl-agreement").value;
            const overrideDx = document.getElementById("hitl-override-dx").value;

            const formData = new FormData();
            formData.append("image_name", activeImgName);
            formData.append("predicted_class", activeTopClass);
            formData.append("attention_focus_rating", rating);
            formData.append("clinician_agreement", agreement);
            formData.append("clinician_diagnosis", overrideDx);
            formData.append("notes", `Calibration via Web UI - Agreement: ${agreement}, Localization: ${rating}`);

            try {
                const res = await fetch("/api/clinician-feedback", { method: "POST", body: formData });
                if (!res.ok) throw new Error("Logging failed");
                if (hitlSuccessMsg) {
                    hitlSuccessMsg.classList.remove("hidden");
                    setTimeout(() => hitlSuccessMsg.classList.add("hidden"), 4000);
                }
            } catch (err) {
                alert(`Feedback logging error: ${err.message}`);
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.innerHTML = '<i class="fa-solid fa-stamp"></i> Record Calibration Audit';
            }
        });
    }

});
