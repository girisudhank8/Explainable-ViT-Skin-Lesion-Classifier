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
                b.classList.remove("active", "text-blue-600");
                b.classList.add("text-slate-500");
            });
            btn.classList.add("active", "text-blue-600");
            btn.classList.remove("text-slate-500");

            tabContents.forEach(content => {
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
    btnDiagnose.addEventListener("click", async () => {
        btnDiagnose.disabled = true;
        btnDiagnose.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing ViT Attention & All Modules...';

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

        try {
            // Run Diagnosis, Benchmark, and Layer-Wise Depth simultaneously via Promise.all
            const [diagRes] = await Promise.all([
                fetch("/api/diagnose", { method: "POST", body: formData }),
                runBenchmark(),
                runLayerwise()
            ]);

            if (!diagRes.ok) {
                const errData = await diagRes.json();
                throw new Error(errData.detail || "Diagnosis failed");
            }

            const data = await diagRes.json();
            renderDiagnosisResults(data);
        } catch (err) {
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

});
