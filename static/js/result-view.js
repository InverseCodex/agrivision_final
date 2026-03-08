const featuresPanel = document.querySelector(".features-panel");
const toggleAdvancedResultsButton = document.getElementById("toggle-advanced-results");

const recommendationList = document.getElementById("ai-recommendation-list");
const progressLabel = document.getElementById("rec-progress");
const notesInput = document.getElementById("farmer-notes");
const saveNotesButton = document.getElementById("save-notes-btn");

const savePersonalizationButton = document.getElementById("save-personalization-btn");
const personalizationStatus = document.getElementById("personalization-status");
const customTitleInput = document.getElementById("custom-title-input");
const fieldNameInput = document.getElementById("field-name-input");
const cropTypeInput = document.getElementById("crop-type-input");

const cropSourceImage = document.getElementById("crop-source-image");
const cropOverlayCanvas = document.getElementById("crop-overlay-canvas");
const cropDrawFrame = document.getElementById("crop-draw-frame");
const cropAnalysisLayout = document.getElementById("crop-analysis-layout");
const resetCropShapeButton = document.getElementById("reset-crop-shape");
const clearCropResultsButton = document.getElementById("clear-crop-results");
const runCroppedAnalysisButton = document.getElementById("run-cropped-analysis");
const showCropToolButton = document.getElementById("show-crop-tool");
const cropAnalysisStatus = document.getElementById("crop-analysis-status");

const croppedOriginalPreview = document.getElementById("cropped-original-preview");
const croppedHeatzonePreview = document.getElementById("cropped-heatzone-preview");
const segmentGridButtons = document.getElementById("segment-grid-buttons");
const segmentVisualGrid = document.getElementById("segment-visual-grid");
const segmentDetailCard = document.getElementById("segment-detail-card");

const segBand = document.getElementById("seg-band");
const segScore = document.getElementById("seg-score");
const segGreen = document.getElementById("seg-green");
const segStress = document.getElementById("seg-stress");
const segVigor = document.getElementById("seg-vigor");
const segIssue = document.getElementById("seg-issue");
const advGreenCoverage = document.getElementById("adv-green-coverage");
const advStressZone = document.getElementById("adv-stress-zone");
const advVigorScore = document.getElementById("adv-vigor-score");
const advNextScan = document.getElementById("adv-next-scan");
const advIdxVari = document.getElementById("adv-idx-vari");
const advIdxGli = document.getElementById("adv-idx-gli");
const advIdxNgrdi = document.getElementById("adv-idx-ngrdi");
const advIdxExg = document.getElementById("adv-idx-exg");
const advIdxTgi = document.getElementById("adv-idx-tgi");
const advIdxMgrvi = document.getElementById("adv-idx-mgrvi");
const advIdxLai = document.getElementById("adv-idx-lai");
const advManagementZone = document.getElementById("adv-management-zone");
const advManagementAction = document.getElementById("adv-management-action");
const advCanopyCover = document.getElementById("adv-canopy-cover");
const advBiomass = document.getElementById("adv-biomass");
const advUniformity = document.getElementById("adv-uniformity");
const advYieldPotential = document.getElementById("adv-yield-potential");
const recSegId = document.getElementById("rec-seg-id");
const recSegBand = document.getElementById("rec-seg-band");
const recSegScore = document.getElementById("rec-seg-score");
const recSegGreen = document.getElementById("rec-seg-green");
const recSegStress = document.getElementById("rec-seg-stress");
const recSegVigor = document.getElementById("rec-seg-vigor");
const recSegIssue = document.getElementById("rec-seg-issue");
const recSegReco = document.getElementById("rec-seg-reco");

const initialPersonalizationEl = document.getElementById("initial-personalization");
const savedTheme = localStorage.getItem("site_theme") || "default";
if (["default", "neon", "minimalist"].includes(savedTheme)) {
    document.body.setAttribute("data-theme", savedTheme);
}

const resultId =
    recommendationList?.dataset.resultId ||
    window.location.pathname.split("/").filter(Boolean).pop() ||
    "unknown";

let saveTimer = null;
let segmentRows = 4;
let segmentCols = 4;
let activeSegmentId = "";
let latestSegments = [];

let handlePoints = [];
let activeHandleIndex = -1;

let personalizationState = {
    title: "",
    field_name: "",
    crop_type: "",
    farmer_notes: "",
    recommendation_checks: [],
    flags: [],
};

try {
    const parsed = JSON.parse(initialPersonalizationEl?.textContent || "{}");
    personalizationState = { ...personalizationState, ...parsed };
} catch (_) {
    // ignore malformed json
}

if (customTitleInput && !customTitleInput.value && personalizationState.title) {
    customTitleInput.value = personalizationState.title;
}
if (fieldNameInput && !fieldNameInput.value && personalizationState.field_name) {
    fieldNameInput.value = personalizationState.field_name;
}
if (cropTypeInput && !cropTypeInput.value && personalizationState.crop_type) {
    cropTypeInput.value = personalizationState.crop_type;
}
if (notesInput && !notesInput.value && personalizationState.farmer_notes) {
    notesInput.value = personalizationState.farmer_notes;
}

function sanitizeLabel(text) {
    return String(text || "").trim().slice(0, 120);
}

function setCropStatus(message, state = "idle") {
    if (!cropAnalysisStatus) return;
    cropAnalysisStatus.textContent = message;
    cropAnalysisStatus.classList.remove("error", "success");
    if (state === "error" || state === "success") {
        cropAnalysisStatus.classList.add(state);
    }
}

function setCropViewMode(mode) {
    if (!cropAnalysisLayout) return;
    cropAnalysisLayout.classList.remove("mode-edit", "mode-results");
    if (mode === "results") {
        cropAnalysisLayout.classList.add("mode-results");
    } else {
        cropAnalysisLayout.classList.add("mode-edit");
    }
}

function setSegmentMetricValues(segment) {
    if (!segBand || !segScore || !segGreen || !segStress || !segVigor || !segIssue) {
        return;
    }
    if (!segment || segment.empty) {
        segIssue.textContent = segment?.possible_issue || "-";
        segBand.textContent = "-";
        segScore.textContent = "-";
        segGreen.textContent = "-";
        segStress.textContent = "-";
        segVigor.textContent = "-";
        if (recSegId) recSegId.textContent = segment?.segment_id || "-";
        if (recSegBand) recSegBand.textContent = "-";
        if (recSegScore) recSegScore.textContent = "-";
        if (recSegGreen) recSegGreen.textContent = "-";
        if (recSegStress) recSegStress.textContent = "-";
        if (recSegVigor) recSegVigor.textContent = "-";
        if (recSegIssue) recSegIssue.textContent = segment?.possible_issue || "-";
        if (recSegReco) recSegReco.textContent = segment?.recommendation || "-";
        return;
    }
    segIssue.textContent = segment.possible_issue || "-";
    segBand.textContent = (segment.health_band || "watch").toUpperCase();
    segScore.textContent = `${segment.health_score}`;
    segGreen.textContent = `${segment.green_coverage_pct}%`;
    segStress.textContent = `${segment.estimated_stress_zone_pct}%`;
    segVigor.textContent = `${segment.vegetation_vigor_score}`;

    if (recSegId) recSegId.textContent = segment.segment_id || "-";
    if (recSegBand) recSegBand.textContent = (segment.health_band || "-").toUpperCase();
    if (recSegScore) recSegScore.textContent = `${segment.health_score ?? "-"}`;
    if (recSegGreen) recSegGreen.textContent = `${segment.green_coverage_pct ?? "-"}%`;
    if (recSegStress) recSegStress.textContent = `${segment.estimated_stress_zone_pct ?? "-"}%`;
    if (recSegVigor) recSegVigor.textContent = `${segment.vegetation_vigor_score ?? "-"}`;
    if (recSegIssue) recSegIssue.textContent = segment.possible_issue || "-";
    if (recSegReco) recSegReco.textContent = segment.recommendation || "-";
}

function setAdvancedPanelFromSegment(segment) {
    if (!segment || segment.empty) {
        return;
    }
    if (advGreenCoverage) advGreenCoverage.textContent = `${segment.green_coverage_pct}%`;
    if (advStressZone) advStressZone.textContent = `${segment.estimated_stress_zone_pct}%`;
    if (advVigorScore) advVigorScore.textContent = `${segment.vegetation_vigor_score}/100`;
    if (advNextScan) advNextScan.textContent = segment.recommendation || "Rescan this segment in 3-5 days.";

    if (advIdxVari) advIdxVari.textContent = `VARI: ${segment.vari ?? "-"}`;
    if (advIdxGli) advIdxGli.textContent = `GLI: ${segment.gli ?? "-"}`;
    if (advIdxNgrdi) advIdxNgrdi.textContent = `NGRDI: ${segment.ngrdi ?? "-"}`;
    if (advIdxExg) advIdxExg.textContent = `ExG: ${segment.exg ?? "-"}`;
    if (advIdxTgi) advIdxTgi.textContent = `TGI: ${segment.tgi ?? "-"}`;
    if (advIdxMgrvi) advIdxMgrvi.textContent = `MGRVI: ${segment.mgrvi ?? "-"}`;
    if (advIdxLai) advIdxLai.textContent = `LAI Proxy: ${segment.lai_proxy ?? "-"}`;

    if (advManagementZone) advManagementZone.textContent = segment.management_zone || "-";
    if (advManagementAction) advManagementAction.textContent = segment.management_action || "-";
    if (advCanopyCover) advCanopyCover.textContent = `${segment.canopy_cover_pct ?? "-"}%`;
    if (advBiomass) advBiomass.textContent = `Biomass: ${segment.relative_biomass_score ?? "-"}`;
    if (advUniformity) advUniformity.textContent = `Uniformity: ${segment.stand_uniformity_score ?? "-"}`;
    if (advYieldPotential) advYieldPotential.textContent = `${segment.relative_yield_potential_pct ?? "-"}%`;
}

function resetDefaultCropShape() {
    handlePoints = [
        { x: 0.2, y: 0.2 },
        { x: 0.82, y: 0.18 },
        { x: 0.84, y: 0.82 },
        { x: 0.18, y: 0.85 },
    ];
    drawCropOverlay();
}

function drawCropOverlay() {
    if (!cropOverlayCanvas) return;
    const ctx = cropOverlayCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, cropOverlayCanvas.width, cropOverlayCanvas.height);
    if (handlePoints.length !== 4) return;

    ctx.beginPath();
    ctx.moveTo(handlePoints[0].x * cropOverlayCanvas.width, handlePoints[0].y * cropOverlayCanvas.height);
    for (let i = 1; i < handlePoints.length; i += 1) {
        ctx.lineTo(handlePoints[i].x * cropOverlayCanvas.width, handlePoints[i].y * cropOverlayCanvas.height);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(31, 168, 122, 0.22)";
    ctx.fill();
    ctx.strokeStyle = "#8CFBD8";
    ctx.lineWidth = 2.2;
    ctx.stroke();

    handlePoints.forEach((point, index) => {
        const x = point.x * cropOverlayCanvas.width;
        const y = point.y * cropOverlayCanvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = index === activeHandleIndex ? "#f6fffb" : "#8CFBD8";
        ctx.fill();
        ctx.strokeStyle = "#134d3a";
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function resizeCropCanvas() {
    if (!cropOverlayCanvas || !cropDrawFrame) return;
    const rect = cropDrawFrame.getBoundingClientRect();
    cropOverlayCanvas.width = Math.max(1, Math.floor(rect.width));
    cropOverlayCanvas.height = Math.max(1, Math.floor(rect.height));
    drawCropOverlay();
}

function pointerToNormalizedPoint(event) {
    if (!cropOverlayCanvas) return null;
    const rect = cropOverlayCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
        x: Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
        y: Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1),
    };
}

function findNearestHandleIndex(point) {
    if (!cropOverlayCanvas || !point) return -1;
    const px = point.x * cropOverlayCanvas.width;
    const py = point.y * cropOverlayCanvas.height;
    let bestIdx = -1;
    let bestDist = Number.POSITIVE_INFINITY;

    handlePoints.forEach((h, idx) => {
        const hx = h.x * cropOverlayCanvas.width;
        const hy = h.y * cropOverlayCanvas.height;
        const d = Math.hypot(px - hx, py - hy);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = idx;
        }
    });

    return bestDist <= 24 ? bestIdx : -1;
}

function renderSegmentButtons(segments, grid) {
    if (!segmentGridButtons) return;
    segmentGridButtons.innerHTML = "";
    if (segmentVisualGrid) segmentVisualGrid.innerHTML = "";

    segmentRows = Number(grid?.rows || 4);
    segmentCols = Number(grid?.cols || 4);
    segmentGridButtons.style.gridTemplateColumns = `repeat(${segmentCols}, minmax(0, 1fr))`;

    if (segmentVisualGrid) {
        segmentVisualGrid.style.gridTemplateColumns = `repeat(${segmentCols}, minmax(0, 1fr))`;
        segmentVisualGrid.style.gridTemplateRows = `repeat(${segmentRows}, minmax(0, 1fr))`;
    }

    const segmentMap = new Map(segments.map((segment) => [segment.segment_id, segment]));
    const selectSegment = (segmentId) => {
        const segment = segmentMap.get(segmentId);
        if (!segment) return;
        activeSegmentId = segmentId;

        segmentGridButtons.querySelectorAll(".segment-btn").forEach((node) => {
            node.classList.toggle("active", node.dataset.segmentId === segmentId);
        });
        segmentVisualGrid?.querySelectorAll(".segment-visual-cell").forEach((node) => {
            node.classList.toggle("active", node.dataset.segmentId === segmentId);
        });

        if (segmentDetailCard) {
            if (segment.empty) {
                segmentDetailCard.textContent = `${segment.segment_id}: no cropped pixels detected in this cell.`;
            } else {
                segmentDetailCard.textContent =
                    `${segment.segment_id} | Band: ${(segment.health_band || "watch").toUpperCase()} | ` +
                    `Score: ${segment.health_score} | ` +
                    `Green: ${segment.green_coverage_pct}% | Stress: ${segment.estimated_stress_zone_pct}% | ` +
                    `Vigor: ${segment.vegetation_vigor_score} | Possible issue: ${segment.possible_issue || "-"} | ` +
                    `Recommendation: ${segment.recommendation || "-"}`;
            }
        }

        setSegmentMetricValues(segment);
        setAdvancedPanelFromSegment(segment);
    };

    segments.forEach((segment, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `segment-btn band-${segment.health_band || "na"}`;
        btn.dataset.segmentId = segment.segment_id;
        btn.textContent = segment.segment_id;
        btn.addEventListener("click", () => selectSegment(segment.segment_id));
        segmentGridButtons.appendChild(btn);

        if (segmentVisualGrid) {
            const cell = document.createElement("button");
            cell.type = "button";
            cell.className = `segment-visual-cell band-${segment.health_band || "na"}`;
            cell.dataset.segmentId = segment.segment_id;
            cell.textContent = segment.segment_id;
            cell.title = `Open ${segment.segment_id} results`;
            cell.addEventListener("click", () => selectSegment(segment.segment_id));
            segmentVisualGrid.appendChild(cell);
        }

        if (idx === 0) {
            activeSegmentId = segment.segment_id;
        }
    });

    if (activeSegmentId) {
        selectSegment(activeSegmentId);
    }
}

async function runCroppedAnalysis() {
    if (!runCroppedAnalysisButton) return;
    if (handlePoints.length !== 4) {
        setCropStatus("Reset crop shape first.", "error");
        return;
    }

    runCroppedAnalysisButton.disabled = true;
    runCroppedAnalysisButton.textContent = "Analyzing...";
    setCropStatus("Analyzing cropped area and segment scores...");

    try {
        const response = await fetch(`/api/results/${encodeURIComponent(resultId)}/crop-rerun`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({
                points: handlePoints,
                grid_rows: segmentRows,
                grid_cols: segmentCols,
            }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || "Cropped analysis failed.");
        }

        if (croppedOriginalPreview) {
            croppedOriginalPreview.src = payload.cropped_original_data_url || "";
        }
        if (croppedHeatzonePreview) {
            croppedHeatzonePreview.src = payload.cropped_heatzone_data_url || "";
        }

        latestSegments = Array.isArray(payload.segments) ? payload.segments : [];
        renderSegmentButtons(latestSegments, payload.grid || { rows: 4, cols: 4 });
        setCropStatus("Done. Click any segment cell to see that segment's own values.", "success");
        setCropViewMode("results");
    } catch (error) {
        setCropStatus(String(error.message || "Cropped analysis failed."), "error");
    } finally {
        runCroppedAnalysisButton.disabled = false;
        runCroppedAnalysisButton.textContent = "Analyze Cropped Area";
    }
}

function updateRecommendationProgress() {
    if (!recommendationList || !progressLabel) return;
    const checks = Array.from(recommendationList.querySelectorAll(".rec-check"));
    const done = checks.filter((c) => c.checked).length;
    const total = checks.length;
    const pct = total ? Math.round((done / total) * 100) : 0;
    progressLabel.textContent = `${done}/${total} actions completed (${pct}%)`;
}

function applyRecommendationChecksFromState() {
    if (!recommendationList) return;
    const checks = Array.from(recommendationList.querySelectorAll(".rec-check"));
    checks.forEach((c, idx) => {
        const checked = Boolean(personalizationState.recommendation_checks?.[idx]);
        c.checked = checked;
        const item = c.closest(".rec-item");
        item?.classList.toggle("done", checked);
        const statusChip = item?.querySelector(".rec-status-chip");
        const actionBtn = item?.querySelector(".rec-action-btn");
        if (statusChip) {
            statusChip.textContent = checked ? "Completed" : "Pending";
        }
        if (actionBtn) {
            actionBtn.textContent = checked ? "Mark as Pending" : "Mark as Done";
        }
    });
    updateRecommendationProgress();
}

function collectCurrentRecommendationChecks() {
    if (!recommendationList) return [];
    return Array.from(recommendationList.querySelectorAll(".rec-check")).map((c) => Boolean(c.checked));
}

function readInputsToState() {
    personalizationState.title = customTitleInput?.value.trim() || "";
    personalizationState.field_name = fieldNameInput?.value.trim() || "";
    personalizationState.crop_type = cropTypeInput?.value.trim() || "";
    personalizationState.farmer_notes = notesInput?.value.trim() || "";
    personalizationState.recommendation_checks = collectCurrentRecommendationChecks();
}

async function persistPersonalization(successText = "Saved.") {
    readInputsToState();
    try {
        const response = await fetch(`/api/results/${encodeURIComponent(resultId)}/customization`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({ personalization: personalizationState }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || "Save failed.");
        }
        personalizationState = { ...personalizationState, ...(payload.personalization || {}) };
        const titleEl = document.getElementById("result-title");
        if (titleEl) {
            titleEl.textContent = personalizationState.title || `Result ${resultId}`;
        }
        if (personalizationStatus) {
            personalizationStatus.textContent = successText;
        }
    } catch (error) {
        if (personalizationStatus) {
            personalizationStatus.textContent = String(error.message || "Unable to save personalization.");
        }
    }
}

function queueAutosave(successText = "Saved.") {
    if (saveTimer) {
        clearTimeout(saveTimer);
    }
    saveTimer = setTimeout(() => {
        persistPersonalization(successText);
    }, 420);
}

if (recommendationList) {
    recommendationList.addEventListener("click", (event) => {
        const btn = event.target;
        if (btn instanceof HTMLButtonElement && btn.classList.contains("rec-action-btn")) {
            const item = btn.closest(".rec-item");
            const input = item?.querySelector(".rec-check");
            if (!(input instanceof HTMLInputElement)) return;
            input.checked = !input.checked;
            applyRecommendationChecksFromState();
            queueAutosave("Recommendation actions updated.");
        }
    });

    recommendationList.addEventListener("change", (event) => {
        const target = event.target;
        if (target instanceof HTMLInputElement && target.classList.contains("rec-check")) {
            applyRecommendationChecksFromState();
            queueAutosave("Recommendation actions updated.");
        }
    });
}

saveNotesButton?.addEventListener("click", async () => {
    await persistPersonalization("Notes saved.");
    if (saveNotesButton) {
        const original = saveNotesButton.textContent;
        saveNotesButton.textContent = "Saved";
        setTimeout(() => {
            saveNotesButton.textContent = original || "Save Notes & Checklist";
        }, 900);
    }
});

savePersonalizationButton?.addEventListener("click", async () => {
    await persistPersonalization("Personalization saved.");
});

[customTitleInput, fieldNameInput, cropTypeInput].forEach((el) => {
    el?.addEventListener("input", () => queueAutosave("Draft saved."));
});

notesInput?.addEventListener("input", () => queueAutosave("Draft saved."));

toggleAdvancedResultsButton?.addEventListener("click", () => {
    if (!featuresPanel) return;
    const visible = featuresPanel.classList.toggle("advanced-results-visible");
    toggleAdvancedResultsButton.setAttribute("aria-expanded", String(visible));
    toggleAdvancedResultsButton.textContent = visible ? "Hide Advance Results" : "Advance Results";
});

if (cropSourceImage && cropOverlayCanvas) {
    if (cropSourceImage.complete) {
        resizeCropCanvas();
        resetDefaultCropShape();
    } else {
        cropSourceImage.addEventListener("load", () => {
            resizeCropCanvas();
            resetDefaultCropShape();
        });
    }

    window.addEventListener("resize", resizeCropCanvas);

    cropOverlayCanvas.addEventListener("pointerdown", (event) => {
        const point = pointerToNormalizedPoint(event);
        if (!point) return;
        const idx = findNearestHandleIndex(point);
        if (idx < 0) return;
        activeHandleIndex = idx;
        cropOverlayCanvas.setPointerCapture(event.pointerId);
        cropOverlayCanvas.style.cursor = "grabbing";
        drawCropOverlay();
    });

    cropOverlayCanvas.addEventListener("pointermove", (event) => {
        if (activeHandleIndex < 0) return;
        const point = pointerToNormalizedPoint(event);
        if (!point) return;
        handlePoints[activeHandleIndex] = point;
        drawCropOverlay();
    });

    cropOverlayCanvas.addEventListener("pointerup", (event) => {
        if (activeHandleIndex >= 0) {
            activeHandleIndex = -1;
            cropOverlayCanvas.style.cursor = "grab";
            cropOverlayCanvas.releasePointerCapture(event.pointerId);
            drawCropOverlay();
            setCropStatus("Crop shape updated. Click Analyze Cropped Area.", "success");
        }
    });

    cropOverlayCanvas.addEventListener("pointerleave", () => {
        if (activeHandleIndex < 0) {
            cropOverlayCanvas.style.cursor = "grab";
        }
    });

    resetCropShapeButton?.addEventListener("click", () => {
        resetDefaultCropShape();
        setCropStatus("Crop shape reset. Drag handles to adjust.");
    });

    clearCropResultsButton?.addEventListener("click", () => {
        latestSegments = [];
        activeSegmentId = "";
        if (segmentGridButtons) segmentGridButtons.innerHTML = "";
        if (segmentVisualGrid) segmentVisualGrid.innerHTML = "";
        if (segmentDetailCard) segmentDetailCard.textContent = "No segment selected yet.";
        if (croppedOriginalPreview) croppedOriginalPreview.removeAttribute("src");
        if (croppedHeatzonePreview) croppedHeatzonePreview.removeAttribute("src");
        setSegmentMetricValues(null);
        setCropStatus("Segment results cleared. Adjust crop and analyze again.");
        setCropViewMode("edit");
    });

    showCropToolButton?.addEventListener("click", () => {
        setCropViewMode("edit");
        setCropStatus("Crop editor shown. Adjust points, then analyze again.");
    });

    runCroppedAnalysisButton?.addEventListener("click", runCroppedAnalysis);
}

applyRecommendationChecksFromState();
setSegmentMetricValues(null);
