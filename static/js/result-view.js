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
const captureAltitudeInput = document.getElementById("capture-altitude-input");
const quickTagDisplay = document.getElementById("quick-tag-display");

const cropSourceImage = document.getElementById("crop-source-image");
const cropOverlayCanvas = document.getElementById("crop-overlay-canvas");
const cropDrawFrame = document.getElementById("crop-draw-frame");
const cropAnalysisLayout = document.getElementById("crop-analysis-layout");
const resetCropShapeButton = document.getElementById("reset-crop-shape");
const clearCropResultsButton = document.getElementById("clear-crop-results");
const runCroppedAnalysisButton = document.getElementById("run-cropped-analysis");
const showCropToolButton = document.getElementById("show-crop-tool");
const toggleFocusModeButton = document.getElementById("toggle-focus-mode");
const cropAnalysisStatus = document.getElementById("crop-analysis-status");
const mobileCropPresets = document.getElementById("mobile-crop-presets");
const segmentVisualFrame = document.getElementById("segment-visual-frame");
const segmentViewOriginalButton = document.getElementById("segment-view-original");
const segmentViewHeatzoneButton = document.getElementById("segment-view-heatzone");
const segmentFocusModal = document.getElementById("segment-focus-modal");
const segmentFocusBackdrop = document.getElementById("segment-focus-backdrop");
const closeFocusModeButton = document.getElementById("close-focus-mode");
const focusSegmentTitle = document.getElementById("focus-segment-title");
const focusSegmentViewLabel = document.getElementById("focus-segment-view-label");
const focusSegmentPreview = document.getElementById("focus-segment-preview");
const focusSegmentPreviewEmpty = document.getElementById("focus-segment-preview-empty");
const focusParameterList = document.getElementById("focus-parameter-list");
const focusRecommendationList = document.getElementById("focus-recommendation-list");

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
const advPredictionProbability = document.getElementById("adv-prediction-probability");
const advHealthFeatureSummary = document.getElementById("adv-health-feature-summary");
const advStageFeatureSummary = document.getElementById("adv-stage-feature-summary");
const advHealthyProbability = document.getElementById("adv-healthy-probability");
const advUnhealthyProbability = document.getElementById("adv-unhealthy-probability");
const advMaturityLabel = document.getElementById("adv-maturity-label");
const advMaturityProbability = document.getElementById("adv-maturity-probability");
const advThreshold = document.getElementById("adv-threshold");
const advModelFindings = document.getElementById("adv-model-findings");
const recSegId = document.getElementById("rec-seg-id");
const recSegIssue = document.getElementById("rec-seg-issue");
const recSegReco = document.getElementById("rec-seg-reco");

const initialPersonalizationEl = document.getElementById("initial-personalization");
const forcedTheme = document.body.dataset.forceTheme || "";
const savedTheme = forcedTheme || localStorage.getItem("site_theme") || "default";
const motionEnabled = localStorage.getItem("site_motion_enabled");
if (["default", "neon", "minimalist"].includes(savedTheme)) {
    document.body.setAttribute("data-theme", savedTheme);
}
if (motionEnabled === "false") {
    document.body.classList.add("reduce-motion");
}

const resultId =
    document.body.dataset.resultId ||
    recommendationList?.dataset.resultId ||
    window.location.pathname.split("/").filter(Boolean).pop() ||
    "unknown";

let saveTimer = null;
let segmentRows = 4;
let segmentCols = 4;
let activeSegmentId = "";
let latestSegments = [];
let latestGrid = { rows: 4, cols: 4, row_weights: [1, 1, 1, 1], col_weights: [1, 1, 1, 1] };

let handlePoints = [];
let activeHandleIndex = -1;
let currentSegmentVisualMode = "heatzone";
let focusModeEnabled = false;
let focusModalOpen = false;
let focusModalCloseTimer = null;

function isEmbeddedResultView() {
    return document.body.classList.contains("embedded-result-body");
}

function notifyParentAdvancedMode(open) {
    if (!isEmbeddedResultView() || window.parent === window) return;
    window.parent.postMessage(
        {
            type: "agrivision:advanced-mode",
            open: Boolean(open),
            resultId,
        },
        window.location.origin,
    );
}

function requestEmbeddedResultRefresh() {
    if (isEmbeddedResultView() && window.parent !== window) {
        window.parent.postMessage(
            {
                type: "agrivision:refresh-upload-result",
                resultId,
            },
            window.location.origin,
        );
        return;
    }
    window.location.reload();
}

function formatStatusMetricValue(value, suffix = "") {
    if (value === null || value === undefined || value === "") {
        return "-";
    }
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (!trimmed) {
            return "-";
        }
        const numeric = Number(trimmed);
        if (!Number.isFinite(numeric)) {
            return trimmed;
        }
        value = numeric;
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return String(value);
    }
    const rounded = Math.round(numeric * 100) / 100;
    if (Number.isInteger(rounded)) {
        return `${rounded}${suffix}`;
    }
    return `${rounded.toFixed(2).replace(/\.?0+$/, "")}${suffix}`;
}

function formatHealthBandLabel(value) {
    const normalized = String(value || "").trim().toLowerCase();
    if (normalized === "healthy") return "Healthy";
    if (normalized === "unhealthy_damaged") return "Needs Attention";
    if (normalized === "mature") return "Likely Mature";
    if (!normalized) return "-";
    return normalized
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function formatPhenologicalStageLabel(segment) {
    if (!segment || segment.empty) {
        return "Not scored";
    }
    const growthStageLabel = String(segment.growth_stage_label || "").trim();
    if (growthStageLabel) {
        return growthStageLabel;
    }
    const healthBand = String(segment.health_band || "").trim().toLowerCase();
    const maturityProbability = Number(segment.maturity_probability ?? segment.growth_stage_probability ?? 0);
    if (healthBand === "mature" || maturityProbability >= 0.5) {
        return "Likely mature stage";
    }
    if (maturityProbability > 0) {
        return "Not yet mature stage";
    }
    return "Stage not available";
}

function buildSegmentStatusItems(segment) {
    if (!segment) {
        return null;
    }
    if (segment.empty) {
        return [
            { label: "Phenological Stage", value: "Not scored" },
            { label: "Healthiness", value: "No usable crop" },
            { label: "Canopy Cover", value: "-" },
            { label: "Vigor", value: "-" },
            { label: "Stand Uniformity", value: "-" },
            { label: "Green Coverage", value: "-" },
        ];
    }
    return [
        { label: "Phenological Stage", value: formatPhenologicalStageLabel(segment) },
        { label: "Healthiness", value: formatHealthBandLabel(segment.health_band) },
        { label: "Canopy Cover", value: formatStatusMetricValue(segment.canopy_cover_pct, "%") },
        { label: "Vigor", value: formatStatusMetricValue(segment.vegetation_vigor_score, "/100") },
        { label: "Stand Uniformity", value: formatStatusMetricValue(segment.stand_uniformity_score, "/100") },
        { label: "Green Coverage", value: formatStatusMetricValue(segment.green_coverage_pct, "%") },
    ];
}

function notifyParentSegmentStatus(segment = null) {
    if (!isEmbeddedResultView() || window.parent === window) return;
    window.parent.postMessage(
        {
            type: "agrivision:segment-status",
            resultId,
            status: buildSegmentStatusItems(segment),
            segmentId: segment?.segment_id || "",
        },
        window.location.origin,
    );
}

function isCoarsePointer() {
    return window.matchMedia("(pointer: coarse)").matches || window.innerWidth <= 780;
}

let personalizationState = {
    title: "",
    field_name: "",
    crop_type: "",
    capture_altitude_m: "",
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

const SEGMENT_STATUS_BANDS = [
    { minRatio: 1.2, label: "Healthy", severity: "very-high" },
    { minRatio: 1.0, label: "Likely Healthy", severity: "high" },
    { minRatio: 0.85, label: "Likely Stressed", severity: "normal" },
    { minRatio: 0.7, label: "Stressed", severity: "low" },
    { minRatio: 0, label: "Need Attention", severity: "very-low" },
];

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
if (captureAltitudeInput && !captureAltitudeInput.value && personalizationState.capture_altitude_m) {
    captureAltitudeInput.value = personalizationState.capture_altitude_m;
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

function formatFeatureSummaryBlock(featureNames, featureValues) {
    const names = Array.isArray(featureNames) ? featureNames : [];
    const values = featureValues && typeof featureValues === "object" ? featureValues : {};
    if (!names.length) {
        return "<p>-</p>";
    }
    return names.map((name) => {
        const label = String(name || "").replaceAll("_", " ").toUpperCase();
        const value = values[name] ?? "-";
        return `<p>${label}: ${value}</p>`;
    }).join("");
}

function formatFeatureSummaryInline(featureNames, featureValues) {
    const names = Array.isArray(featureNames) ? featureNames : [];
    const values = featureValues && typeof featureValues === "object" ? featureValues : {};
    if (!names.length) {
        return "-";
    }
    return names.map((name) => `${String(name || "").replaceAll("_", " ").toUpperCase()}: ${values[name] ?? "-"}`).join(", ");
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
    const ratio = value / Math.max(threshold, 1);
    const band = SEGMENT_STATUS_BANDS.find((item) => ratio >= item.minRatio);
    return band?.label || "Need Attention";
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
    const ratio = value / Math.max(threshold, 1);
    const band = SEGMENT_STATUS_BANDS.find((item) => ratio >= item.minRatio);
    return band?.severity || "very-low";
}

function getAdaptiveSegmentRecommendation(segment) {
    if (!segment || segment.empty) {
        return segment?.recommendation || "-";
    }
    return segment.recommendation || "The model likely sees this segment as stable, but it is still best to verify suspicious areas in person.";
}

function getSegmentDamageClass(segment) {
    if (!segment || segment.empty) return "empty";
    const damageProbability = Number(segment.unhealthy_damaged_probability ?? segment.prediction_probability ?? 0);
    if (damageProbability >= 0.8) return "critical";
    if (damageProbability >= 0.5) return "likely";
    return "healthy";
}

function formatPercentFromProbability(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return `${(Number(value) * 100).toFixed(1)}%`;
}

function getActiveSegment() {
    return latestSegments.find((segment) => segment.segment_id === activeSegmentId) || null;
}

function getSegmentBoundsPercent(segment, grid) {
    const colWeights = Array.isArray(grid?.col_weights) && grid.col_weights.length
        ? grid.col_weights.map((value) => Math.max(1, Number(value) || 1))
        : Array.from({ length: Number(grid?.cols || segmentCols) || 1 }, () => 1);
    const rowWeights = Array.isArray(grid?.row_weights) && grid.row_weights.length
        ? grid.row_weights.map((value) => Math.max(1, Number(value) || 1))
        : Array.from({ length: Number(grid?.rows || segmentRows) || 1 }, () => 1);

    const colIndex = Math.max(0, Number(segment.col || 1) - 1);
    const rowIndex = Math.max(0, Number(segment.row || 1) - 1);
    const totalCols = colWeights.reduce((sum, value) => sum + value, 0);
    const totalRows = rowWeights.reduce((sum, value) => sum + value, 0);
    const left = (colWeights.slice(0, colIndex).reduce((sum, value) => sum + value, 0) / totalCols) * 100;
    const top = (rowWeights.slice(0, rowIndex).reduce((sum, value) => sum + value, 0) / totalRows) * 100;
    const width = ((colWeights[colIndex] || 1) / totalCols) * 100;
    const height = ((rowWeights[rowIndex] || 1) / totalRows) * 100;
    return { left, top, width, height };
}

function renderFocusSegmentPreview(segment) {
    if (!focusSegmentPreview || !focusSegmentPreviewEmpty) return;
    if (focusSegmentViewLabel) {
        focusSegmentViewLabel.textContent = `Current mode: ${currentSegmentVisualMode === "original" ? "Original" : "AI Image"}`;
    }

    if (!segment) {
        focusSegmentPreview.style.backgroundImage = "none";
        focusSegmentPreview.style.backgroundSize = "";
        focusSegmentPreview.style.backgroundPosition = "";
        focusSegmentPreviewEmpty.hidden = false;
        return;
    }

    const source = currentSegmentVisualMode === "original"
        ? (croppedOriginalPreview?.src || "")
        : (croppedHeatzonePreview?.src || "");
    if (!source) {
        focusSegmentPreview.style.backgroundImage = "none";
        focusSegmentPreviewEmpty.hidden = false;
        return;
    }

    const bounds = getSegmentBoundsPercent(segment, latestGrid);
    const safeWidth = Math.max(bounds.width, 0.5);
    const safeHeight = Math.max(bounds.height, 0.5);
    focusSegmentPreview.style.backgroundImage = `url("${source}")`;
    focusSegmentPreview.style.backgroundSize = `${100 / (safeWidth / 100)}% ${100 / (safeHeight / 100)}%`;
    focusSegmentPreview.style.backgroundPosition = `${(bounds.left / Math.max(100 - safeWidth, 0.0001)) * 100}% ${(bounds.top / Math.max(100 - safeHeight, 0.0001)) * 100}%`;
    focusSegmentPreviewEmpty.hidden = true;
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
        if (quickTagDisplay) quickTagDisplay.textContent = segment?.empty ? "EMPTY" : "Select a segment";
        if (recSegId) recSegId.textContent = segment?.segment_id || "-";
        if (recSegReco) recSegReco.textContent = getAdaptiveSegmentRecommendation(segment);
        if (segDetailId) segDetailId.textContent = segment?.segment_id || "-";
        if (segDetailScore) segDetailScore.textContent = segment?.empty ? "EMPTY" : "-";
        if (segDetailVigor) segDetailVigor.textContent = "-";
        if (segDetailCanopy) segDetailCanopy.textContent = "-";
        if (segDetailBiomass) segDetailBiomass.textContent = "-";
        if (segDetailUniformity) segDetailUniformity.textContent = "-";
        if (segDetailGreen) segDetailGreen.textContent = "-";
        if (segDetailStress) segDetailStress.textContent = "-";
        renderFocusModeDetails(null);
        return;
    }
    if (quickTagDisplay) {
        quickTagDisplay.textContent = String(segment.health_band || "unknown").replaceAll("_", " ").toUpperCase();
    }
    if (recSegId) recSegId.textContent = segment.segment_id || "-";
    if (recSegReco) recSegReco.textContent = getAdaptiveSegmentRecommendation(segment);
    if (segDetailId) segDetailId.textContent = segment.segment_id || "-";
    if (segDetailScore) {
        segDetailScore.textContent = segment.health_band === "mature" ? "MATURE" : formatScore(segment.health_score);
    }
    if (segDetailVigor) segDetailVigor.textContent = formatScore(segment.vegetation_vigor_score);
    if (segDetailCanopy) segDetailCanopy.textContent = formatScore(segment.canopy_cover_pct, "%");
    if (segDetailBiomass) segDetailBiomass.textContent = formatScore(segment.relative_biomass_score);
    if (segDetailUniformity) segDetailUniformity.textContent = formatScore(segment.stand_uniformity_score);
    if (segDetailGreen) segDetailGreen.textContent = formatScore(segment.green_coverage_pct, "%");
    if (segDetailStress) segDetailStress.textContent = formatScore(segment.estimated_stress_zone_pct, "%");
    renderFocusModeDetails(segment);
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
    if (advPredictionProbability) advPredictionProbability.textContent = formatPercentFromProbability(segment.prediction_probability);
    if (advHealthFeatureSummary) advHealthFeatureSummary.innerHTML = formatFeatureSummaryBlock(segment.health_feature_names, segment.health_feature_values);
    if (advStageFeatureSummary) advStageFeatureSummary.innerHTML = formatFeatureSummaryBlock(segment.stage_feature_names, segment.stage_feature_values);
    if (advHealthyProbability) advHealthyProbability.textContent = formatPercentFromProbability(segment.healthy_probability);
    if (advUnhealthyProbability) advUnhealthyProbability.textContent = formatPercentFromProbability(segment.unhealthy_damaged_probability);
    if (advMaturityLabel) advMaturityLabel.textContent = String(segment.growth_stage_label || "-").replaceAll("_", " ").toUpperCase();
    if (advMaturityProbability) advMaturityProbability.textContent = formatPercentFromProbability(segment.growth_stage_probability);
    if (advThreshold) advThreshold.textContent = formatScore(segment.threshold ?? 0.5);
    if (advModelFindings) {
        const label = String(segment.health_band || "").replaceAll("_", " ").toUpperCase();
        const stage = String(segment.growth_stage_label || "unknown stage").replaceAll("_", " ");
        advModelFindings.textContent = `Selected segment prediction is likely ${label || "UNKNOWN"} while the stage model points to ${stage}.`;
    }
}

function setSegmentVisualMode(mode) {
    currentSegmentVisualMode = mode === "original" ? "original" : "heatzone";
    if (segmentVisualFrame) {
        segmentVisualFrame.classList.toggle("is-original", currentSegmentVisualMode === "original");
        segmentVisualFrame.classList.toggle("is-heatzone", currentSegmentVisualMode !== "original");
    }
    segmentViewOriginalButton?.classList.toggle("viewer-btn-active", currentSegmentVisualMode === "original");
    segmentViewHeatzoneButton?.classList.toggle("viewer-btn-active", currentSegmentVisualMode !== "original");
    if (focusModalOpen) {
        renderFocusSegmentPreview(getActiveSegment());
    }
    normalizeSegmentPreviewFrame();
}

function setFocusMode(enabled) {
    focusModeEnabled = Boolean(enabled);
    toggleFocusModeButton?.classList.toggle("viewer-btn-active", focusModeEnabled);
    toggleFocusModeButton?.setAttribute("aria-pressed", String(focusModeEnabled));
    toggleFocusModeButton.textContent = focusModeEnabled ? "Advanced Mode On" : "Advanced Mode";
    closeFocusModal();
}

function openFocusModal() {
    if (!segmentFocusModal) return;
    if (focusModalCloseTimer) {
        clearTimeout(focusModalCloseTimer);
        focusModalCloseTimer = null;
    }
    focusModalOpen = true;
    segmentFocusModal.classList.remove("is-closing");
    segmentFocusModal.hidden = false;
    segmentFocusModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("focus-modal-open");
    notifyParentAdvancedMode(true);
}

function closeFocusModal() {
    if (!segmentFocusModal) return;
    focusModalOpen = false;
    segmentFocusModal.classList.add("is-closing");
    segmentFocusModal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("focus-modal-open");
    notifyParentAdvancedMode(false);
    if (focusModalCloseTimer) {
        clearTimeout(focusModalCloseTimer);
    }
    focusModalCloseTimer = setTimeout(() => {
        if (!segmentFocusModal) return;
        segmentFocusModal.hidden = true;
        segmentFocusModal.classList.remove("is-closing");
        focusModalCloseTimer = null;
    }, 190);
}

function fitFrameToViewport(frameEl, imgEl, options = {}) {
    if (!isEmbeddedResultView() || !frameEl || !imgEl) return;
    const naturalWidth = Number(imgEl.naturalWidth || 0);
    const naturalHeight = Number(imgEl.naturalHeight || 0);
    if (!naturalWidth || !naturalHeight) return;

    const minHeight = Number(options.minHeight || 260);
    const viewportOffset = Number(options.viewportOffset || 240);
    const parentWidth = Math.max(
        1,
        Math.floor(frameEl.parentElement?.clientWidth || frameEl.clientWidth || naturalWidth),
    );
    const maxHeight = Math.max(minHeight, window.innerHeight - viewportOffset);

    let width = parentWidth;
    let height = width * (naturalHeight / naturalWidth);

    if (height > maxHeight) {
        height = maxHeight;
        width = height * (naturalWidth / naturalHeight);
    }

    frameEl.style.width = `${Math.round(Math.min(width, parentWidth))}px`;
    frameEl.style.height = `${Math.round(height)}px`;
    frameEl.style.maxWidth = "100%";
    frameEl.style.marginInline = "auto";

    imgEl.style.width = "100%";
    imgEl.style.height = "100%";
    imgEl.style.objectFit = "contain";
}

function getContainedMediaRect(frameEl, mediaEl) {
    if (!frameEl || !mediaEl) return null;
    const frameWidth = Number(frameEl.clientWidth || 0);
    const frameHeight = Number(frameEl.clientHeight || 0);
    const naturalWidth = Number(mediaEl.naturalWidth || 0);
    const naturalHeight = Number(mediaEl.naturalHeight || 0);
    if (!frameWidth || !frameHeight || !naturalWidth || !naturalHeight) {
        return null;
    }

    const frameRatio = frameWidth / frameHeight;
    const mediaRatio = naturalWidth / naturalHeight;

    let width = frameWidth;
    let height = frameHeight;
    let left = 0;
    let top = 0;

    if (mediaRatio > frameRatio) {
        width = frameWidth;
        height = width / mediaRatio;
        top = (frameHeight - height) / 2;
    } else {
        height = frameHeight;
        width = height * mediaRatio;
        left = (frameWidth - width) / 2;
    }

    return { left, top, width, height };
}

function syncOverlayToContainedMedia(overlayEl, frameEl, mediaEl) {
    if (!overlayEl || !frameEl || !mediaEl) return;
    const rect = getContainedMediaRect(frameEl, mediaEl);
    if (!rect) return;
    overlayEl.style.left = `${rect.left}px`;
    overlayEl.style.top = `${rect.top}px`;
    overlayEl.style.width = `${rect.width}px`;
    overlayEl.style.height = `${rect.height}px`;
}

function renderFocusModeDetails(segment) {
    if (!focusSegmentTitle || !focusParameterList || !focusRecommendationList) return;

    if (!segment || segment.empty) {
        focusSegmentTitle.textContent = "Select a segment to inspect it closely.";
        renderFocusSegmentPreview(null);
        focusParameterList.innerHTML = "<p>No segment selected.</p>";
        focusRecommendationList.innerHTML = "<li>Select a segment to load guidance.</li>";
        return;
    }

    const severityLabel = getSegmentDamageClass(segment).replace("-", " ").toUpperCase();
    focusSegmentTitle.textContent = `Segment ${segment.segment_id} - ${severityLabel}`;
    renderFocusSegmentPreview(segment);
    const parameters = [
        ["Health score", segment.health_band === "mature" ? "MATURE" : formatScore(segment.health_score)],
        ["Prediction confidence", formatPercentFromProbability(segment.confidence)],
        ["Healthy probability", formatPercentFromProbability(segment.healthy_probability)],
        ["Unhealthy/damaged probability", formatPercentFromProbability(segment.unhealthy_damaged_probability)],
        ["Growth stage", String(segment.growth_stage_label || "-")],
        ["Growth-stage probability", formatPercentFromProbability(segment.growth_stage_probability)],
        ["Mature-stage probability", formatPercentFromProbability(segment.maturity_probability)],
        ["Green coverage", formatScore(segment.green_coverage_pct, "%")],
        ["Stress zone", formatScore(segment.estimated_stress_zone_pct, "%")],
        ["Vigor", formatScore(segment.vegetation_vigor_score)],
        ["Canopy cover", formatScore(segment.canopy_cover_pct, "%")],
        ["Biomass", formatScore(segment.relative_biomass_score)],
        ["Uniformity", formatScore(segment.stand_uniformity_score)],
        ["Health features", formatFeatureSummaryInline(segment.health_feature_names, segment.health_feature_values)],
        ["Stage features", formatFeatureSummaryInline(segment.stage_feature_names, segment.stage_feature_values)],
        ["Management zone", segment.management_zone || "-"],
        ["Management action", segment.management_action || "-"],
    ];
    focusParameterList.innerHTML = parameters
        .map(([label, value]) => `<div class="focus-parameter-row"><span>${label}</span><strong>${value}</strong></div>`)
        .join("");

    const recommendations = [
        segment.recommendation,
        segment.possible_issue,
        `Model finding: likely ${String(segment.health_band || "").replaceAll("_", " ")} with confidence ${formatPercentFromProbability(segment.confidence)} while the stage model points to ${segment.growth_stage_label || "an unknown stage"}.`,
        segment.unhealthy_damaged_probability >= 0.5
            ? "Action note: verify the flagged area on-site before applying targeted treatment."
            : "Action note: maintain monitoring and compare this segment with surrounding cells on the next scan.",
    ].filter(Boolean);
    focusRecommendationList.innerHTML = recommendations.map((item) => `<li>${item}</li>`).join("");
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
    if (cropSourceImage && cropSourceImage.complete) {
        fitFrameToViewport(cropDrawFrame, cropSourceImage, {
            minHeight: 280,
            viewportOffset: 230,
        });
    }
    syncOverlayToContainedMedia(cropOverlayCanvas, cropDrawFrame, cropSourceImage);
    const rect = cropOverlayCanvas.getBoundingClientRect();
    cropOverlayCanvas.width = Math.max(1, Math.floor(rect.width));
    cropOverlayCanvas.height = Math.max(1, Math.floor(rect.height));
    drawCropOverlay();
}

function normalizeSegmentPreviewFrame() {
    if (!segmentVisualFrame) return;
    const candidate =
        (croppedHeatzonePreview && croppedHeatzonePreview.complete && croppedHeatzonePreview.naturalWidth > 0 && croppedHeatzonePreview.src
            ? croppedHeatzonePreview
            : null) ||
        (croppedOriginalPreview && croppedOriginalPreview.complete && croppedOriginalPreview.naturalWidth > 0 && croppedOriginalPreview.src
            ? croppedOriginalPreview
            : null);
    if (!candidate) return;
    fitFrameToViewport(segmentVisualFrame, candidate, {
        minHeight: 260,
        viewportOffset: 320,
    });
    syncSegmentVisualGridToImage();
}

function getActiveSegmentPreviewImage() {
    if (currentSegmentVisualMode === "original") {
        if (croppedOriginalPreview?.src && croppedOriginalPreview.naturalWidth > 0) return croppedOriginalPreview;
        if (croppedHeatzonePreview?.src && croppedHeatzonePreview.naturalWidth > 0) return croppedHeatzonePreview;
        return null;
    }
    if (croppedHeatzonePreview?.src && croppedHeatzonePreview.naturalWidth > 0) return croppedHeatzonePreview;
    if (croppedOriginalPreview?.src && croppedOriginalPreview.naturalWidth > 0) return croppedOriginalPreview;
    return null;
}

function syncSegmentVisualGridToImage() {
    if (!segmentVisualGrid || !segmentVisualFrame) return;
    const imageEl = getActiveSegmentPreviewImage();
    if (!imageEl) {
        segmentVisualGrid.style.left = "";
        segmentVisualGrid.style.top = "";
        segmentVisualGrid.style.width = "";
        segmentVisualGrid.style.height = "";
        return;
    }
    syncOverlayToContainedMedia(segmentVisualGrid, segmentVisualFrame, imageEl);
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
    latestGrid = grid || latestGrid;

    if (segmentVisualGrid) {
        const colWeights = Array.isArray(grid?.col_weights) && grid.col_weights.length === segmentCols
            ? grid.col_weights
            : Array.from({ length: segmentCols }, () => 1);
        const rowWeights = Array.isArray(grid?.row_weights) && grid.row_weights.length === segmentRows
            ? grid.row_weights
            : Array.from({ length: segmentRows }, () => 1);
        segmentVisualGrid.style.gridTemplateColumns = colWeights.map((weight) => `minmax(0, ${Math.max(1, Number(weight) || 1)}fr)`).join(" ");
        segmentVisualGrid.style.gridTemplateRows = rowWeights.map((weight) => `minmax(0, ${Math.max(1, Number(weight) || 1)}fr)`).join(" ");
        syncSegmentVisualGridToImage();
    }

    const segmentMap = new Map(segments.map((segment) => [segment.segment_id, segment]));
    if (activeSegmentId && !segmentMap.has(activeSegmentId)) {
        activeSegmentId = "";
    }
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
        setAdvancedPanelFromSegment(segment);
        notifyParentSegmentStatus(segment);
        if (focusModeEnabled) {
            openFocusModal();
        }
    };

    const buildSummaryRow = (segment, mode = "summary") => {
        const row = document.createElement("tr");
        row.className = "segment-summary-row";
        row.dataset.segmentId = segment.segment_id;
        if (segment.empty) {
            row.classList.add("segment-summary-row-empty");
        }
        row.tabIndex = 0;
        row.addEventListener("click", () => selectSegment(segment.segment_id));
        row.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                selectSegment(segment.segment_id);
            }
        });

        const values = [
            segment.empty ? `${segment.segment_id} (EMPTY)` : segment.segment_id,
            segment.empty
                ? "EMPTY"
                : (
                    segment.health_band === "mature"
                        ? "MATURE"
                        : (mode === "summary" ? classifySegmentMetric(segment, "health_score") : formatScore(segment.health_score))
                ),
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
            cell.className = `segment-visual-cell damage-${getSegmentDamageClass(segment)}`;
            cell.dataset.segmentId = segment.segment_id;
            cell.style.gridColumn = String(Number(segment.col || 1));
            cell.style.gridRow = String(Number(segment.row || 1));
            cell.textContent = segment.segment_id;
            if (segment.empty) {
                cell.classList.add("segment-visual-cell-empty");
                cell.title = `${segment.segment_id}: empty`;
                cell.addEventListener("click", () => selectSegment(segment.segment_id));
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
        notifyParentSegmentStatus(null);
    }
    syncSegmentVisualGridToImage();
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
        window.setTimeout(normalizeSegmentPreviewFrame, 50);
        window.setTimeout(normalizeSegmentPreviewFrame, 180);

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
    personalizationState.capture_altitude_m = captureAltitudeInput?.value || "";
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

[customTitleInput, fieldNameInput, cropTypeInput, captureAltitudeInput].forEach((el) => {
    el?.addEventListener("input", () => queueAutosave("Draft saved."));
    el?.addEventListener("change", () => queueAutosave("Draft saved."));
});

notesInput?.addEventListener("input", () => queueAutosave("Draft saved."));

toggleAdvancedResultsButton?.addEventListener("click", () => {
    if (!featuresPanel) return;
    const visible = featuresPanel.classList.toggle("advanced-results-visible");
    toggleAdvancedResultsButton.setAttribute("aria-expanded", String(visible));
    toggleAdvancedResultsButton.textContent = visible ? "Hide Advanced Results" : "Show Advanced Results";
});

segmentViewOriginalButton?.addEventListener("click", () => setSegmentVisualMode("original"));
segmentViewHeatzoneButton?.addEventListener("click", () => setSegmentVisualMode("heatzone"));
toggleFocusModeButton?.addEventListener("click", () => setFocusMode(!focusModeEnabled));
closeFocusModeButton?.addEventListener("click", () => closeFocusModal());
segmentFocusBackdrop?.addEventListener("click", () => closeFocusModal());
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && focusModalOpen) {
        closeFocusModal();
    }
});
setSegmentVisualMode("heatzone");
setFocusMode(false);

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

    window.addEventListener("resize", () => {
        resizeCropCanvas();
        normalizeSegmentPreviewFrame();
    });

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
        notifyParentSegmentStatus(null);
        setCropStatus("Segment results cleared. Adjust crop and analyze again.");
        setCropViewMode("edit");
    });

    showCropToolButton?.addEventListener("click", () => {
        if (isEmbeddedResultView()) {
            closeFocusModal();
            setCropStatus("Refreshing the current upload...");
            requestEmbeddedResultRefresh();
            return;
        }
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

[croppedOriginalPreview, croppedHeatzonePreview].forEach((img) => {
    img?.addEventListener("load", normalizeSegmentPreviewFrame);
});

applyRecommendationChecksFromState();
setSegmentMetricValues(null);
