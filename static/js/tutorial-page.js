(() => {
    const SETTINGS_KEYS = {
        theme: "site_theme",
        motionEnabled: "site_motion_enabled"
    };

    const stepLabel = document.getElementById("tutorial-step-label");
    const stepTitle = document.getElementById("tutorial-step-title");
    const stepBody = document.getElementById("tutorial-step-body");
    const guideHeading = document.getElementById("tutorial-guide-heading");
    const guideBadge = document.getElementById("tutorial-guide-badge");
    const guideCopy = document.getElementById("tutorial-guide-copy");
    const tipList = document.getElementById("tutorial-tip-list");
    const progressFill = document.getElementById("tutorial-progress-fill");
    const prevButton = document.getElementById("tutorial-prev");
    const nextButton = document.getElementById("tutorial-next");
    const stageChips = Array.from(document.querySelectorAll(".tutorial-stage-chip"));
    const windows = {
        upload: document.getElementById("scene-upload"),
        results: document.getElementById("scene-results"),
        resultView: document.getElementById("scene-result-view")
    };

    if (
        !stepLabel ||
        !stepTitle ||
        !stepBody ||
        !guideHeading ||
        !guideBadge ||
        !guideCopy ||
        !tipList ||
        !progressFill ||
        !prevButton ||
        !nextButton ||
        !stageChips.length
    ) {
        return;
    }

    const steps = [
        {
            windowKey: "upload",
            target: '[data-tutorial-target="upload-dropzone"]',
            badge: "Upload Window",
            title: "Open the Upload Image window",
            body: "This guided page starts where the real website starts. Users first enter the Upload Image window to begin a field scan.",
            tips: [
                "This duplicated window mirrors the real Upload Image tab layout.",
                "The drop zone is the first interaction point in the workflow.",
                "Users can leave the tutorial anytime with the Exit Tutorial button."
            ]
        },
        {
            windowKey: "upload",
            target: '[data-tutorial-target="upload-preview"]',
            badge: "Upload Window",
            title: "Choose an image and check the preview",
            body: "After selecting a file, the preview panel shows the exact field image that will be analyzed.",
            tips: [
                "The preview helps prevent uploading the wrong image.",
                "This step teaches users to confirm the image before continuing.",
                "The visual matches the behavior of the real upload page."
            ]
        },
        {
            windowKey: "upload",
            target: '[data-tutorial-target="upload-submit"]',
            badge: "Upload Window",
            title: "Send the image to analysis",
            body: "Once the image and file details look correct, the Upload Image button sends it into the processing flow.",
            tips: [
                "The file info card confirms the selected image name and size.",
                "Clear resets the selection if the wrong file was chosen.",
                "Upload Image is the action that starts the real analysis pipeline."
            ]
        },
        {
            windowKey: "results",
            target: '[data-tutorial-target="results-filters"]',
            badge: "View Results",
            title: "Open the View Results window",
            body: "After analysis finishes, users move to View Results to search, filter, and sort completed scans.",
            tips: [
                "Search is useful when many result cards already exist.",
                "Band filter narrows the list by crop health level.",
                "Sort helps users review newest or strongest results first."
            ]
        },
        {
            windowKey: "results",
            target: '[data-tutorial-target="results-card"]',
            badge: "View Results",
            title: "Open a result card",
            body: "Each result card shows a quick summary and gives access to the detailed result page or downloadable PDF.",
            tips: [
                "Open Result leads to the full result-view page.",
                "The preview image helps users identify the correct scan fast.",
                "The health band gives a fast visual summary before opening details."
            ]
        },
        {
            windowKey: "resultView",
            target: '[data-tutorial-target="result-crop"]',
            badge: "Result View",
            title: "Learn Crop & Segment Analysis",
            body: "This section teaches users how to shape the crop area, analyze only the selected region, and inspect smaller field segments.",
            tips: [
                "Crop & Segment Analysis is where the user focuses on the farm area that matters.",
                "Specific Segment Recommendation updates based on the selected segment.",
                "This is the best place to localize a problem inside a larger field image."
            ]
        },
        {
            windowKey: "resultView",
            target: '[data-tutorial-target="result-recommendations"]',
            badge: "Result View",
            title: "Understand Results and Recommendation",
            body: "This section turns analysis into field actions. Users can review tasks, track completion, and keep notes about what has already been done.",
            tips: [
                "Action Progress shows how many recommended tasks are already complete.",
                "Recommendation cards translate AI output into practical next steps.",
                "The notes box lets users save work history tied to this result."
            ]
        },
        {
            windowKey: "resultView",
            target: '[data-tutorial-target="result-advanced"]',
            badge: "Result View",
            title: "Review Useful RGB Features and Advance Results",
            body: "The final tutorial step explains the detailed metrics area. This is where users find summary features, segment tables, technical values, and management guidance.",
            tips: [
                "Useful RGB Features presents the fastest high-level interpretation of the result.",
                "Advance Results reveals deeper segment tables and technical measurement values.",
                "Result Personalization helps rename and organize scans for easier future review."
            ]
        }
    ];

    let currentStep = 0;

    function applySavedAppearance() {
        const theme = localStorage.getItem(SETTINGS_KEYS.theme) || "default";
        const motionEnabled = localStorage.getItem(SETTINGS_KEYS.motionEnabled);
        document.body.setAttribute("data-theme", theme);
        document.body.classList.toggle("reduce-motion", motionEnabled === "false");
    }

    function clearHighlights() {
        Object.values(windows).forEach((windowEl) => {
            windowEl?.classList.remove("is-highlighted");
        });
        document.querySelectorAll(".tutorial-card.is-highlighted").forEach((card) => {
            card.classList.remove("is-highlighted");
        });
    }

    function renderTips(tips) {
        tipList.innerHTML = "";
        tips.forEach((tip) => {
            const item = document.createElement("div");
            item.className = "tutorial-tip";
            item.textContent = tip;
            tipList.appendChild(item);
        });
    }

    function renderStep(stepIndex) {
        const step = steps[stepIndex];
        if (!step) {
            return;
        }

        currentStep = stepIndex;
        clearHighlights();

        const windowEl = windows[step.windowKey];
        const targetEl = document.querySelector(step.target);
        windowEl?.classList.add("is-highlighted");
        targetEl?.classList.add("is-highlighted");
        targetEl?.scrollIntoView({ behavior: document.body.classList.contains("reduce-motion") ? "auto" : "smooth", block: "center" });

        stepLabel.textContent = `Step ${stepIndex + 1} of ${steps.length}`;
        stepTitle.textContent = step.title;
        stepBody.textContent = step.body;
        guideHeading.textContent = step.title;
        guideBadge.textContent = step.badge;
        guideCopy.textContent = step.body;
        progressFill.style.width = `${((stepIndex + 1) / steps.length) * 100}%`;
        prevButton.disabled = stepIndex === 0;
        nextButton.textContent = stepIndex === steps.length - 1 ? "Finish" : "Next";
        renderTips(step.tips);

        stageChips.forEach((chip, index) => {
            chip.classList.toggle("is-active", index === stepIndex);
        });
    }

    prevButton.addEventListener("click", () => {
        if (currentStep > 0) {
            renderStep(currentStep - 1);
        }
    });

    nextButton.addEventListener("click", () => {
        if (currentStep >= steps.length - 1) {
            window.location.href = "/?window=upload";
            return;
        }
        renderStep(currentStep + 1);
    });

    stageChips.forEach((chip, index) => {
        chip.addEventListener("click", () => {
            renderStep(index);
        });
    });

    applySavedAppearance();
    renderStep(0);
})();
