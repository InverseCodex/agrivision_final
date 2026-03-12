const featuresPanel = document.querySelector(".features-panel");
const toggleAdvancedResultsButton = document.getElementById("toggle-advanced-results");
const pdfDownloadLink = document.querySelector(".pdf-btn");

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
const mobileCropPresets = document.getElementById("mobile-crop-presets");

const croppedOriginalPreview = document.getElementById("cropped-original-preview");
const croppedHeatzonePreview = document.getElementById("cropped-heatzone-preview");
const segmentVisualGrid = document.getElementById("segment-visual-grid");
const segmentSummaryTableBody = document.getElementById("segment-summary-table-body");
const segDetailId = document.getElementById("seg-detail-id");
const segDetailScore = document.getElementById("seg-detail-score");
const segDetailVigor = document.getElementById("seg-detail-vigor");
const segDetailCanopy = document.getElementById("seg-detail-canopy");
const segDetailBiomass = document.getElementById("seg-detail-biomass");
const segDetailUniformity = document.getElementById("seg-detail-uniformity");
const segDetailGreen = document.getElementById("seg-detail-green");
const segDetailStress = document.getElementById("seg-detail-stress");
const advSegmentSummaryTableBody = document.getElementById("adv-segment-summary-table-body");
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

function isCoarsePointer() {
    return window.matchMedia("(pointer: coarse)").matches || window.innerWidth <= 780;
}

let personalizationState = {
    title: "",
    field_name: "",
    crop_type: "",
    farmer_notes: "",
    recommendation_checks: [],
    flags: [],
};

const SEGMENT_THRESHOLDS = {
    health_score: 60,
    vegetation_vigor_score: 60,
    canopy_cover_pct: 40,
    relative_biomass_score: 45,
    stand_uniformity_score: 55,
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

function formatScore(value, suffix = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${value}${suffix}`;
}

function classifySegmentMetric(segment, key) {
    if (!segment || segment.empty) {
        return "N/A";
    }
    const value = Number(segment[key]);
    const threshold = Number(SEGMENT_THRESHOLDS[key]);
    if (Number.isNaN(value) || Number.isNaN(threshold)) {
        return "N/A";
    }

    if (value <= threshold * 0.75) return "Very Low";
    if (value <= threshold * 0.95) return "Low";
    if (value <= threshold * 1.1) return "Normal";
    if (value <= threshold * 1.25) return "High";
    return "Very High";
}

function getMetricSeverity(segment, key) {
    if (!segment || segment.empty) {
        return "na";
    }
    const value = Number(segment[key]);
    const threshold = Number(SEGMENT_THRESHOLDS[key]);
    if (Number.isNaN(value) || Number.isNaN(threshold)) {
        return "na";
    }
    if (value <= threshold * 0.75) return "very-low";
    if (value <= threshold * 0.95) return "low";
    if (value <= threshold * 1.1) return "normal";
    if (value <= threshold * 1.25) return "high";
    return "very-high";
}

function getAdaptiveSegmentRecommendation(segment) {
    if (!segment || segment.empty) {
        return segment?.recommendation || "-";
    }

    const checks = [
        {
            label: "health score",
            below: Number(segment.health_score) < SEGMENT_THRESHOLDS.health_score,
            message: "Health is low in this segment. Prioritize a field check and confirm whether crop stress is spreading.",
        },
        {
            label: "vigor",
            below: Number(segment.vegetation_vigor_score) < SEGMENT_THRESHOLDS.vegetation_vigor_score,
            message: "Vigor is low in this segment. Review nutrient delivery and irrigation timing to help plant growth recover.",
        },
        {
            label: "canopy",
            below: Number(segment.canopy_cover_pct) < SEGMENT_THRESHOLDS.canopy_cover_pct,
            message: "Canopy is low in this segment. Inspect for sparse stand, delayed growth, or missing plants in this section.",
        },
        {
            label: "biomass",
            below: Number(segment.relative_biomass_score) < SEGMENT_THRESHOLDS.relative_biomass_score,
            message: "Biomass is low in this segment. Check whether plants are undersized and adjust feeding or water support as needed.",
        },
        {
            label: "uniformity",
            below: Number(segment.stand_uniformity_score) < SEGMENT_THRESHOLDS.stand_uniformity_score,
            message: "Uniformity is low in this segment. Compare weak and strong patches for irrigation inconsistency, pests, or uneven emergence.",
        },
    ];

    const flagged = checks.filter((item) => item.below);
    if (!flagged.length) {
        return segment.recommendation || "All tracked segment metrics are above threshold. Maintain the current program and monitor on the next scan.";
    }
    if (flagged.length === 1) {
        return flagged[0].message;
    }
    return `${flagged[0].message} Also monitor ${flagged.slice(1).map((item) => item.label).join(", ")} because they are also below threshold in this segment.`;
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
    if (!segment || segment.empty) {
        if (recSegId) recSegId.textContent = segment?.segment_id || "-";
        if (recSegReco) recSegReco.textContent = getAdaptiveSegmentRecommendation(segment);
        if (segDetailId) segDetailId.textContent = segment?.segment_id || "-";
        if (segDetailScore) segDetailScore.textContent = "-";
        if (segDetailVigor) segDetailVigor.textContent = "-";
        if (segDetailCanopy) segDetailCanopy.textContent = "-";
        if (segDetailBiomass) segDetailBiomass.textContent = "-";
        if (segDetailUniformity) segDetailUniformity.textContent = "-";
        if (segDetailGreen) segDetailGreen.textContent = "-";
        if (segDetailStress) segDetailStress.textContent = "-";
        return;
    }
    if (recSegId) recSegId.textContent = segment.segment_id || "-";
    if (recSegReco) recSegReco.textContent = getAdaptiveSegmentRecommendation(segment);
    if (segDetailId) segDetailId.textContent = segment.segment_id || "-";
    if (segDetailScore) segDetailScore.textContent = formatScore(segment.health_score);
    if (segDetailVigor) segDetailVigor.textContent = formatScore(segment.vegetation_vigor_score);
    if (segDetailCanopy) segDetailCanopy.textContent = formatScore(segment.canopy_cover_pct, "%");
    if (segDetailBiomass) segDetailBiomass.textContent = formatScore(segment.relative_biomass_score);
    if (segDetailUniformity) segDetailUniformity.textContent = formatScore(segment.stand_uniformity_score);
    if (segDetailGreen) segDetailGreen.textContent = formatScore(segment.green_coverage_pct, "%");
    if (segDetailStress) segDetailStress.textContent = formatScore(segment.estimated_stress_zone_pct, "%");
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

    const handleRadius = isCoarsePointer() ? 12 : 8;
    handlePoints.forEach((point, index) => {
        const x = point.x * cropOverlayCanvas.width;
        const y = point.y * cropOverlayCanvas.height;
        ctx.beginPath();
        ctx.arc(x, y, handleRadius, 0, Math.PI * 2);
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

    const threshold = isCoarsePointer() ? 44 : 24;
    return bestDist <= threshold ? bestIdx : -1;
}

function applyCropPreset(preset) {
    const presets = {
        full: [
            { x: 0.04, y: 0.04 },
            { x: 0.96, y: 0.04 },
            { x: 0.96, y: 0.96 },
            { x: 0.04, y: 0.96 },
        ],
        center: [
            { x: 0.18, y: 0.18 },
            { x: 0.82, y: 0.18 },
            { x: 0.82, y: 0.82 },
            { x: 0.18, y: 0.82 },
        ],
        top: [
            { x: 0.08, y: 0.06 },
            { x: 0.92, y: 0.06 },
            { x: 0.92, y: 0.54 },
            { x: 0.08, y: 0.54 },
        ],
        bottom: [
            { x: 0.08, y: 0.46 },
            { x: 0.92, y: 0.46 },
            { x: 0.92, y: 0.94 },
            { x: 0.08, y: 0.94 },
        ],
        left: [
            { x: 0.06, y: 0.08 },
            { x: 0.54, y: 0.08 },
            { x: 0.54, y: 0.92 },
            { x: 0.06, y: 0.92 },
        ],
        right: [
            { x: 0.46, y: 0.08 },
            { x: 0.94, y: 0.08 },
            { x: 0.94, y: 0.92 },
            { x: 0.46, y: 0.92 },
        ],
    };
    const points = presets[preset];
    if (!points) return;
    handlePoints = points.map((p) => ({ ...p }));
    drawCropOverlay();
    setCropStatus(`Preset applied: ${preset}. Tap Analyze Cropped Area.`, "success");
}

function renderSegmentButtons(segments, grid) {
    if (segmentSummaryTableBody) {
        segmentSummaryTableBody.innerHTML = "";
    }
    if (advSegmentSummaryTableBody) {
        advSegmentSummaryTableBody.innerHTML = "";
    }
    if (segmentVisualGrid) {
        segmentVisualGrid.innerHTML = "";
    }

    segmentRows = Number(grid?.rows || 4);
    segmentCols = Number(grid?.cols || 4);

    if (segmentVisualGrid) {
        segmentVisualGrid.style.gridTemplateColumns = `repeat(${segmentCols}, minmax(0, 1fr))`;
        segmentVisualGrid.style.gridTemplateRows = `repeat(${segmentRows}, minmax(0, 1fr))`;
    }

    const segmentMap = new Map(segments.map((segment) => [segment.segment_id, segment]));
    const selectSegment = (segmentId) => {
        const segment = segmentMap.get(segmentId);
        if (!segment) return;
        activeSegmentId = segmentId;

        document.querySelectorAll(".segment-summary-row").forEach((node) => {
            node.classList.toggle("active", node.dataset.segmentId === segmentId);
        });
        document.querySelectorAll(".segment-visual-cell").forEach((node) => {
            node.classList.toggle("active", node.dataset.segmentId === segmentId);
        });

        setSegmentMetricValues(segment);
    };

    const buildSummaryRow = (segment, mode = "summary") => {
        const row = document.createElement("tr");
        row.className = "segment-summary-row";
        row.dataset.segmentId = segment.segment_id;
        if (segment.empty) {
            row.classList.add("segment-summary-row-empty");
        } else {
            row.tabIndex = 0;
            row.addEventListener("click", () => selectSegment(segment.segment_id));
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectSegment(segment.segment_id);
                }
            });
        }

        const values = [
            segment.empty ? `${segment.segment_id} (N/A)` : segment.segment_id,
            mode === "summary" ? classifySegmentMetric(segment, "health_score") : formatScore(segment.health_score),
            mode === "summary" ? classifySegmentMetric(segment, "vegetation_vigor_score") : formatScore(segment.vegetation_vigor_score),
            mode === "summary" ? classifySegmentMetric(segment, "canopy_cover_pct") : formatScore(segment.canopy_cover_pct, "%"),
            mode === "summary" ? classifySegmentMetric(segment, "relative_biomass_score") : formatScore(segment.relative_biomass_score),
            mode === "summary" ? classifySegmentMetric(segment, "stand_uniformity_score") : formatScore(segment.stand_uniformity_score),
        ];

        const severityKeys = [null, "health_score", "vegetation_vigor_score", "canopy_cover_pct", "relative_biomass_score", "stand_uniformity_score"];
        values.forEach((value, index) => {
            const cell = document.createElement("td");
            cell.textContent = value;
            if (mode === "summary" && severityKeys[index]) {
                cell.classList.add(`metric-${getMetricSeverity(segment, severityKeys[index])}`);
            }
            row.appendChild(cell);
        });

        return row;
    };

    segments.forEach((segment) => {
        if (segmentVisualGrid) {
            const cell = document.createElement("button");
            cell.type = "button";
            cell.className = `segment-visual-cell band-${segment.health_band || "na"}`;
            cell.dataset.segmentId = segment.segment_id;
            cell.textContent = segment.empty ? "" : segment.segment_id;
            if (segment.empty) {
                cell.classList.add("segment-visual-cell-empty");
                cell.title = `${segment.segment_id}: outside selected crop area`;
                cell.disabled = true;
            } else {
                cell.title = segment.segment_id;
                cell.addEventListener("click", () => selectSegment(segment.segment_id));
            }
            segmentVisualGrid.appendChild(cell);
        }

        if (segmentSummaryTableBody) {
            segmentSummaryTableBody.appendChild(buildSummaryRow(segment, "summary"));
        }
        if (advSegmentSummaryTableBody) {
            advSegmentSummaryTableBody.appendChild(buildSummaryRow(segment, "detailed"));
        }

        if (!segment.empty && !activeSegmentId) {
            activeSegmentId = segment.segment_id;
        }
    });

    if (!segments.length && segmentSummaryTableBody) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = "No analyzable segment in the selected crop area.";
        row.appendChild(cell);
        segmentSummaryTableBody.appendChild(row);
    }
    if (!segments.length && advSegmentSummaryTableBody) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.textContent = "No segment results yet.";
        row.appendChild(cell);
        advSegmentSummaryTableBody.appendChild(row);
    }

    if (activeSegmentId) {
        selectSegment(activeSegmentId);
    } else {
        setSegmentMetricValues(null);
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
        setCropStatus("Done. Review the segment table below and select a row to update the recommendation.", "success");
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

if (pdfDownloadLink) {
    const pdfUrl = new URL(pdfDownloadLink.href, window.location.origin);
    pdfUrl.searchParams.set("theme", savedTheme);
    pdfDownloadLink.href = pdfUrl.pathname + pdfUrl.search;
}

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
        let idx = findNearestHandleIndex(point);
        if (idx < 0 && isCoarsePointer() && handlePoints.length) {
            idx = handlePoints
                .map((h, i) => ({ i, d: Math.hypot(point.x - h.x, point.y - h.y) }))
                .sort((a, b) => a.d - b.d)[0]?.i ?? -1;
        }
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
        if (segmentSummaryTableBody) {
            segmentSummaryTableBody.innerHTML = '<tr><td colspan="6">No segment results yet.</td></tr>';
        }
        if (advSegmentSummaryTableBody) {
            advSegmentSummaryTableBody.innerHTML = '<tr><td colspan="6">No segment results yet.</td></tr>';
        }
        if (segmentVisualGrid) segmentVisualGrid.innerHTML = "";
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

    mobileCropPresets?.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLButtonElement)) return;
        const preset = target.dataset.cropPreset;
        if (!preset) return;
        applyCropPreset(preset);
    });
}

applyRecommendationChecksFromState();
setSegmentMetricValues(null);
