(() => {
    const stateElement = document.getElementById("dashboard-state");
    let initialState = { initialWindow: "home", initialResultId: "" };

    try {
        initialState = {
            ...initialState,
            ...(JSON.parse(stateElement?.textContent || "{}")),
        };
    } catch (_) {
        // Ignore malformed embedded state.
    }

    const body = document.body;
    const mobileNavToggle = document.getElementById("mobile-nav-toggle");
    const sidebarVisibilityToggle = document.getElementById("sidebar-visibility-toggle");
    const sidebar = document.getElementById("dashboard-sidebar");
    const navButtons = Array.from(document.querySelectorAll(".nav-button"));
    const contentWindows = Array.from(document.querySelectorAll(".content-window"));
    const desktopNavBreakpoint = 920;
    const sidebarHiddenStorageKey = "agrivision_sidebar_hidden";

    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("image-upload");
    const uploadDropzone = document.getElementById("upload-dropzone");
    const clearFileButton = document.getElementById("clear-file");
    const uploadSubmitButton = document.getElementById("upload-submit");
    const uploadStatus = document.getElementById("upload-status");
    const fileNameLabel = document.getElementById("file-name");
    const fileSizeLabel = document.getElementById("file-size");
    const previewImage = document.getElementById("preview-image");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const uploadLayout = document.querySelector(".upload-layout");
    const uploadStage = document.querySelector(".upload-stage");
    const uploadFormState = document.getElementById("upload-form-state");
    const uploadResultState = document.getElementById("upload-result-state");
    const uploadStageTitle = document.getElementById("upload-stage-title");
    const uploadStateChip = document.getElementById("upload-state-chip");

    const resultEmbedFrame = document.getElementById("result-embed-frame");
    const statusList = document.getElementById("status-list");
    const recommendationList = document.getElementById("recommendation-list");
    const recommendationContext = document.getElementById("recommendation-context");
    const recommendationSegmentChip = document.getElementById("recommendation-segment-chip");
    const backToUploadButton = document.getElementById("back-to-upload");

    const historyGrid = document.getElementById("history-grid");
    const historySearch = document.getElementById("history-search");
    const historyBandFilter = document.getElementById("history-band-filter");
    const historySort = document.getElementById("history-sort");

    let currentResult = null;
    let currentWindowName = "home";
    let frameMutationObserver = null;
    let frameResizeObserver = null;
    const recommendationEmptyText = "Open a result to load segment recommendations here.";
    const recommendationPendingText = "Segmentation is running. Recommendations will appear after a segment is selected.";

    function getEmbeddedResultMinimumHeight() {
        return window.innerWidth <= 640 ? 360 : 420;
    }

    function setEmbeddedAdvancedMode(open) {
        const active = Boolean(open);
        body.classList.toggle("embedded-advanced-open", active);
        uploadStage?.classList.toggle("embedded-advanced-open", active);
        if (active) {
            closeMobileNav();
        }
        window.requestAnimationFrame(resizeEmbeddedResultFrame);
    }

    function refreshCurrentUploadResult(resultId = "") {
        const nextResultId = resultId || currentResult?.image_id || "";
        const url = new URL(window.location.href);
        url.searchParams.set("window", "upload");
        if (nextResultId) {
            url.searchParams.set("result", nextResultId);
        } else {
            url.searchParams.delete("result");
        }
        window.history.replaceState({}, "", url);
        window.location.reload();
    }

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function closeMobileNav() {
        body.classList.remove("nav-open");
        mobileNavToggle?.setAttribute("aria-expanded", "false");
    }

    function toggleMobileNav() {
        const isOpen = body.classList.toggle("nav-open");
        mobileNavToggle?.setAttribute("aria-expanded", String(isOpen));
    }

    function isDesktopViewport() {
        return window.innerWidth > desktopNavBreakpoint;
    }

    function updateSidebarVisibilityButton() {
        if (!sidebarVisibilityToggle) return;
        const hidden = isDesktopViewport() && body.classList.contains("sidebar-hidden");
        sidebarVisibilityToggle.textContent = hidden ? ">" : "<";
        sidebarVisibilityToggle.setAttribute("aria-expanded", String(!hidden));
        sidebarVisibilityToggle.setAttribute("aria-label", hidden ? "Show navbar" : "Hide navbar");
        sidebarVisibilityToggle.setAttribute("title", hidden ? "Show navbar" : "Hide navbar");
    }

    function setDesktopSidebarHidden(hidden, persistPreference = true) {
        const nextHidden = Boolean(hidden) && isDesktopViewport();
        body.classList.toggle("sidebar-hidden", nextHidden);
        if (persistPreference) {
            try {
                localStorage.setItem(sidebarHiddenStorageKey, String(nextHidden));
            } catch (_) {
                // Ignore storage failures.
            }
        }
        updateSidebarVisibilityButton();
    }

    function restoreDesktopSidebarPreference() {
        if (!isDesktopViewport()) {
            body.classList.remove("sidebar-hidden");
            updateSidebarVisibilityButton();
            return;
        }

        let preferredHidden = false;
        try {
            preferredHidden = localStorage.getItem(sidebarHiddenStorageKey) === "true";
        } catch (_) {
            preferredHidden = false;
        }
        setDesktopSidebarHidden(preferredHidden, false);
    }

    function updateUrl(windowName, resultId = "") {
        const url = new URL(window.location.href);
        url.searchParams.set("window", windowName);
        if (windowName === "upload" && resultId) {
            url.searchParams.set("result", resultId);
        } else {
            url.searchParams.delete("result");
        }
        window.history.replaceState({}, "", url);
    }

    function activateWindow(windowName, preferredButtonId = "") {
        currentWindowName = windowName;

        navButtons.forEach((button) => {
            const targetWindow = button.dataset.windowTarget;
            const shouldActivate = preferredButtonId
                ? button.id === preferredButtonId
                : targetWindow === windowName && button.id === `${windowName}-button`;
            button.classList.toggle("is-active", shouldActivate);
        });

        contentWindows.forEach((contentWindow) => {
            contentWindow.classList.toggle("is-active", contentWindow.id === `${windowName}-window`);
        });

        updateUrl(windowName, currentResult?.image_id || "");
        closeMobileNav();
    }

    function setUploadStatus(message, isError = false) {
        if (!uploadStatus) return;
        uploadStatus.textContent = message;
        uploadStatus.style.color = isError ? "var(--danger)" : "";
    }

    function clearPreview() {
        if (previewImage) {
            previewImage.removeAttribute("src");
            previewImage.hidden = true;
        }
        previewPlaceholder?.removeAttribute("hidden");
    }

    function resetFileSelection() {
        if (fileInput) {
            fileInput.value = "";
        }
        if (fileNameLabel) {
            fileNameLabel.textContent = "No image selected";
        }
        if (fileSizeLabel) {
            fileSizeLabel.textContent = "-";
        }
        if (uploadSubmitButton) {
            uploadSubmitButton.disabled = true;
        }
        clearPreview();
        setUploadStatus("Select an image to begin analysis.");
    }

    function formatFileSize(size) {
        if (!Number.isFinite(size) || size <= 0) return "-";
        if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
        return `${(size / (1024 * 1024)).toFixed(2)} MB`;
    }

    function handleSelectedFile(file) {
        if (!file) {
            resetFileSelection();
            return;
        }

        fileNameLabel.textContent = file.name;
        fileSizeLabel.textContent = formatFileSize(file.size);
        uploadSubmitButton.disabled = false;
        setUploadStatus("Image ready. Analyze when you are ready.");

        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = String(event.target?.result || "");
            previewImage.hidden = false;
            previewPlaceholder?.setAttribute("hidden", "hidden");
        };
        reader.readAsDataURL(file);
    }

    function showUploadState() {
        currentResult = null;
        setEmbeddedAdvancedMode(false);
        uploadLayout?.classList.remove("is-result-mode");
        uploadStage?.classList.remove("is-result-mode");
        uploadFormState.hidden = false;
        uploadFormState.removeAttribute("hidden");
        uploadResultState.hidden = true;
        uploadResultState.setAttribute("hidden", "hidden");
        uploadStageTitle.textContent = "Upload Image";
        uploadStateChip.textContent = "Waiting for image";
        if (resultEmbedFrame) {
            resultEmbedFrame.removeAttribute("src");
            resultEmbedFrame.style.height = "";
        }
        disconnectEmbeddedFrameObservers();
        renderStatus();
        renderRecommendations();
        updateUrl(currentWindowName === "upload" ? "upload" : currentWindowName);
    }

    function renderStatus(result = null) {
        if (!statusList) return;
        const items = result?.status || [
            { label: "Phenological Stage", value: "-" },
            { label: "Healthiness", value: "-" },
            { label: "Canopy Cover", value: "-" },
            { label: "Vigor", value: "-" },
            { label: "Stand Uniformity", value: "-" },
            { label: "Green Coverage", value: "-" },
        ];

        statusList.innerHTML = items
            .map((item) => `<li class="status-item"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></li>`)
            .join("");
    }

    function renderRecommendations(result = null) {
        if (!recommendationList) return;
        let items = [recommendationEmptyText];
        let contextText = recommendationEmptyText;
        let segmentLabel = "Segment -";

        if (result?.recommendations?.length) {
            items = result.recommendations;
            contextText = result.context || "Stage-aware guidance for the selected segment.";
            segmentLabel = result.segmentId ? `Segment ${result.segmentId}` : segmentLabel;
        } else if (result?.pending) {
            items = [recommendationPendingText];
            contextText = recommendationPendingText;
        }

        if (recommendationContext) {
            recommendationContext.textContent = contextText;
        }
        if (recommendationSegmentChip) {
            recommendationSegmentChip.textContent = segmentLabel;
        }
        recommendationList.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    }

    function disconnectEmbeddedFrameObservers() {
        if (frameMutationObserver) {
            frameMutationObserver.disconnect();
            frameMutationObserver = null;
        }
        if (frameResizeObserver) {
            frameResizeObserver.disconnect();
            frameResizeObserver = null;
        }
    }

    function resizeEmbeddedResultFrame() {
        if (!resultEmbedFrame?.contentDocument) return;
        const doc = resultEmbedFrame.contentDocument;
        const bodyEl = doc.body;
        const docEl = doc.documentElement;
        const nextHeight = Math.max(
            getEmbeddedResultMinimumHeight(),
            bodyEl?.scrollHeight || 0,
            docEl?.scrollHeight || 0,
            bodyEl?.offsetHeight || 0,
            docEl?.offsetHeight || 0,
        );
        resultEmbedFrame.style.height = `${nextHeight}px`;
    }

    function attachEmbeddedResultFrameObservers() {
        if (!resultEmbedFrame?.contentDocument?.body) return;
        disconnectEmbeddedFrameObservers();
        const bodyEl = resultEmbedFrame.contentDocument.body;
        frameMutationObserver = new MutationObserver(() => {
            window.requestAnimationFrame(resizeEmbeddedResultFrame);
        });
        frameMutationObserver.observe(bodyEl, {
            childList: true,
            subtree: true,
            attributes: true,
            characterData: true,
        });

        if ("ResizeObserver" in window) {
            frameResizeObserver = new ResizeObserver(() => {
                window.requestAnimationFrame(resizeEmbeddedResultFrame);
            });
            frameResizeObserver.observe(bodyEl);
        }

        window.setTimeout(resizeEmbeddedResultFrame, 80);
        window.setTimeout(resizeEmbeddedResultFrame, 240);
    }

    function loadEmbeddedResult(resultId) {
        if (!resultEmbedFrame || !resultId) return;
        const nextSrc = `/results/${encodeURIComponent(resultId)}/embedded`;
        if (resultEmbedFrame.dataset.resultId === resultId && resultEmbedFrame.getAttribute("src") === nextSrc) {
            resizeEmbeddedResultFrame();
            return;
        }
        disconnectEmbeddedFrameObservers();
        resultEmbedFrame.dataset.resultId = resultId;
        resultEmbedFrame.src = nextSrc;
    }

    function renderResult(result, activateUploadWindow = true) {
        currentResult = result;
        setEmbeddedAdvancedMode(false);
        uploadLayout?.classList.add("is-result-mode");
        uploadStage?.classList.add("is-result-mode");
        uploadFormState.hidden = true;
        uploadFormState.setAttribute("hidden", "hidden");
        uploadResultState.hidden = false;
        uploadResultState.removeAttribute("hidden");
        uploadStageTitle.textContent = "Image View";
        uploadStateChip.textContent = result.health_band_label || "Viewing result";

        renderStatus(result);
        renderRecommendations({ pending: true });
        loadEmbeddedResult(result.image_id || "");

        if (activateUploadWindow) {
            activateWindow("upload", "upload-button");
        } else {
            updateUrl("upload", result.image_id || "");
        }
    }

    function buildHistoryCard(entry) {
        const card = document.createElement("article");
        card.className = "history-card";
        card.dataset.resultId = entry.image_id || "";
        card.dataset.healthBand = entry.health_band || "";
        card.dataset.healthScore = String(entry.health_score || 0);
        card.dataset.createdAt = entry.created_at || "";

        const preview = entry.heatzone_url
            ? `<img src="${escapeHtml(entry.heatzone_url)}" alt="Result ${escapeHtml(entry.image_id || "")} preview">`
            : `<div class="history-card-fallback">No preview</div>`;

        card.innerHTML = `
            <div class="history-card-media">${preview}</div>
            <div class="history-card-body">
                <div class="history-meta">
                    <span class="history-badge">${escapeHtml(entry.health_band_label || "Result")}</span>
                    <span>${escapeHtml(entry.created_at || "-")}</span>
                </div>
                <h3>${escapeHtml(entry.display_title || `Result ${entry.image_id || ""}`)}</h3>
                <p>${escapeHtml(entry.summary || "Analysis available.")}</p>
                <div class="history-actions">
                    <button class="history-button primary" type="button" data-action="view" data-result-id="${escapeHtml(entry.image_id || "")}">View Result</button>
                    <button class="history-button secondary" type="button" data-action="delete" data-result-id="${escapeHtml(entry.image_id || "")}">Delete</button>
                </div>
            </div>
        `;

        return card;
    }

    function ensureEmptyHistoryState(message) {
        if (!historyGrid) return;
        let emptyState = historyGrid.querySelector(".history-empty");
        if (!emptyState) {
            emptyState = document.createElement("div");
            emptyState.className = "history-empty";
            historyGrid.appendChild(emptyState);
        }
        emptyState.innerHTML = `<h3>No results yet</h3><p>${message}</p>`;
        emptyState.style.display = "block";
    }

    function hideEmptyHistoryState() {
        const emptyState = historyGrid?.querySelector(".history-empty");
        if (emptyState) {
            emptyState.style.display = "none";
        }
    }

    function upsertHistoryCard(entry) {
        if (!historyGrid || !entry?.image_id) return;
        const existing = historyGrid.querySelector(`.history-card[data-result-id="${entry.image_id}"]`);
        const replacement = buildHistoryCard(entry);
        if (existing) {
            existing.replaceWith(replacement);
        } else {
            const emptyState = historyGrid.querySelector(".history-empty");
            if (emptyState) {
                emptyState.remove();
            }
            historyGrid.prepend(replacement);
        }
        applyHistoryFilters();
    }

    function applyHistoryFilters() {
        if (!historyGrid) return;
        const cards = Array.from(historyGrid.querySelectorAll(".history-card"));
        if (!cards.length) {
            ensureEmptyHistoryState("Upload and analyze an image to build your history list.");
            return;
        }

        const query = (historySearch?.value || "").trim().toLowerCase();
        const selectedBand = (historyBandFilter?.value || "all").toLowerCase();
        const sortMode = historySort?.value || "newest";

        cards.forEach((card) => {
            const resultId = (card.dataset.resultId || "").toLowerCase();
            const band = (card.dataset.healthBand || "").toLowerCase();
            const matchesQuery = !query || resultId.includes(query);
            const matchesBand = selectedBand === "all" || band === selectedBand;
            card.style.display = matchesQuery && matchesBand ? "" : "none";
        });

        const visibleCards = cards.filter((card) => card.style.display !== "none");
        visibleCards.sort((cardA, cardB) => {
            const scoreA = Number(cardA.dataset.healthScore || 0);
            const scoreB = Number(cardB.dataset.healthScore || 0);
            const timeA = Date.parse(cardA.dataset.createdAt || "") || 0;
            const timeB = Date.parse(cardB.dataset.createdAt || "") || 0;

            if (sortMode === "oldest") return timeA - timeB;
            if (sortMode === "score_desc") return scoreB - scoreA;
            if (sortMode === "score_asc") return scoreA - scoreB;
            return timeB - timeA;
        });

        visibleCards.forEach((card) => historyGrid.appendChild(card));

        if (!visibleCards.length) {
            ensureEmptyHistoryState("No matching history items for the current filters.");
        } else {
            hideEmptyHistoryState();
        }
    }

    async function fetchResultSummary(resultId) {
        const response = await fetch(`/api/results/${encodeURIComponent(resultId)}/summary`, {
            headers: {
                Accept: "application/json",
            },
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload.error || "Unable to load result.");
        }
        return payload.result;
    }

    async function handleUploadSubmit(event) {
        event.preventDefault();
        if (!uploadForm) return;

        const formData = new FormData(uploadForm);
        uploadSubmitButton.disabled = true;
        setUploadStatus("Analyzing image. Please wait...");

        try {
            const response = await fetch("/upload-image", {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "X-Requested-With": "fetch",
                },
                body: formData,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || "Upload failed.");
            }

            renderResult(payload.result, true);
            upsertHistoryCard(payload.history_entry);
            uploadForm.reset();
            resetFileSelection();
            setUploadStatus("Analysis complete.");
        } catch (error) {
            uploadSubmitButton.disabled = false;
            setUploadStatus(error.message || "Upload failed.", true);
        }
    }

    async function handleHistoryAction(event) {
        const target = event.target.closest("[data-action]");
        if (!(target instanceof HTMLElement)) return;

        const action = target.dataset.action;
        const resultId = target.dataset.resultId;
        if (!action || !resultId) return;

        if (action === "view") {
            const originalText = target.textContent;
            target.textContent = "Opening...";
            target.setAttribute("disabled", "true");

            try {
                const result = await fetchResultSummary(resultId);
                renderResult(result, true);
            } catch (error) {
                window.alert(error.message || "Unable to open this result.");
            } finally {
                target.textContent = originalText || "View Result";
                target.removeAttribute("disabled");
            }
            return;
        }

        if (action === "delete") {
            const confirmed = window.confirm(`Delete result ${resultId}? This cannot be undone.`);
            if (!confirmed) return;

            const originalText = target.textContent;
            target.textContent = "Deleting...";
            target.setAttribute("disabled", "true");

            try {
                const response = await fetch(`/api/results/${encodeURIComponent(resultId)}`, {
                    method: "DELETE",
                    headers: {
                        Accept: "application/json",
                    },
                });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.error || "Delete failed.");
                }

                target.closest(".history-card")?.remove();
                if (currentResult?.image_id === resultId) {
                    showUploadState();
                }
                applyHistoryFilters();
            } catch (error) {
                window.alert(error.message || "Delete failed.");
                target.textContent = originalText || "Delete";
                target.removeAttribute("disabled");
            }
        }
    }

    navButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const windowName = button.dataset.windowTarget;
            if (!windowName) return;
            activateWindow(windowName, button.id);
        });
    });

    mobileNavToggle?.addEventListener("click", toggleMobileNav);
    sidebarVisibilityToggle?.addEventListener("click", () => {
        setDesktopSidebarHidden(!body.classList.contains("sidebar-hidden"));
    });

    document.addEventListener("click", (event) => {
        if (window.innerWidth > desktopNavBreakpoint || !body.classList.contains("nav-open")) return;
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (sidebar?.contains(target) || mobileNavToggle?.contains(target)) return;
        closeMobileNav();
    });

    window.addEventListener("message", (event) => {
        if (event.origin !== window.location.origin) return;
        const data = event.data;
        if (!data?.type) return;
        if (data.type === "agrivision:advanced-mode") {
            setEmbeddedAdvancedMode(Boolean(data.open));
            return;
        }
        if (data.type === "agrivision:return-to-upload") {
            showUploadState();
            return;
        }
        if (data.type === "agrivision:segment-status") {
            const messageResultId = String(data.resultId || "");
            if (!currentResult?.image_id || currentResult.image_id !== messageResultId) {
                return;
            }
            if (Array.isArray(data.status) && data.status.length) {
                renderStatus({ status: data.status });
            } else {
                renderStatus(currentResult);
            }
            if (Array.isArray(data.recommendations) && data.recommendations.length) {
                renderRecommendations({
                    recommendations: data.recommendations,
                    context: String(data.possibleIssue || ""),
                    segmentId: String(data.segmentId || ""),
                });
            } else {
                renderRecommendations(currentResult ? { pending: true } : null);
            }
            return;
        }
        if (data.type === "agrivision:refresh-upload-result") {
            refreshCurrentUploadResult(String(data.resultId || ""));
        }
    });

    fileInput?.addEventListener("change", () => {
        handleSelectedFile(fileInput.files?.[0] || null);
    });

    uploadDropzone?.addEventListener("click", () => fileInput?.click());
    uploadDropzone?.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            fileInput?.click();
        }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        uploadDropzone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadDropzone.classList.add("is-dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadDropzone?.addEventListener(eventName, (event) => {
            event.preventDefault();
            uploadDropzone.classList.remove("is-dragging");
        });
    });

    uploadDropzone?.addEventListener("drop", (event) => {
        const transfer = event.dataTransfer;
        if (!transfer?.files?.length || !fileInput) return;
        fileInput.files = transfer.files;
        handleSelectedFile(transfer.files[0]);
    });

    clearFileButton?.addEventListener("click", resetFileSelection);
    uploadForm?.addEventListener("submit", handleUploadSubmit);

    backToUploadButton?.addEventListener("click", showUploadState);

    [historySearch, historyBandFilter, historySort].forEach((control) => {
        control?.addEventListener("input", applyHistoryFilters);
        control?.addEventListener("change", applyHistoryFilters);
    });

    historyGrid?.addEventListener("click", handleHistoryAction);

    window.addEventListener("resize", () => {
        if (window.innerWidth > desktopNavBreakpoint) {
            closeMobileNav();
            restoreDesktopSidebarPreference();
        } else {
            body.classList.remove("sidebar-hidden");
            updateSidebarVisibilityButton();
        }
        resizeEmbeddedResultFrame();
    });

    resultEmbedFrame?.addEventListener("load", () => {
        resizeEmbeddedResultFrame();
        attachEmbeddedResultFrameObservers();
    });

    renderStatus();
    renderRecommendations();
    resetFileSelection();
    applyHistoryFilters();
    restoreDesktopSidebarPreference();

    const initialWindow = ["home", "upload", "history"].includes(initialState.initialWindow)
        ? initialState.initialWindow
        : "home";

    if (initialState.initialResultId) {
        fetchResultSummary(initialState.initialResultId)
            .then((result) => renderResult(result, true))
            .catch(() => activateWindow(initialWindow, `${initialWindow}-button`));
    } else {
        activateWindow(initialWindow, initialWindow === "home" ? "home-button" : `${initialWindow}-button`);
    }
})();
