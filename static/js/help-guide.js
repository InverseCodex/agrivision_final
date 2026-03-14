(() => {
    const shell = document.getElementById("tutorial-shell");
    const beginButton = document.getElementById("tutorial-begin");
    const nextButton = document.getElementById("tutorial-next");
    const prevButton = document.getElementById("tutorial-prev");
    const exitTopButton = document.getElementById("tutorial-exit-top");
    const exitBottomButton = document.getElementById("tutorial-exit-bottom");
    const cardKicker = document.getElementById("tutorial-card-kicker");
    const cardTitle = document.getElementById("tutorial-card-title");
    const cardBody = document.getElementById("tutorial-card-body");
    const windowBadge = document.getElementById("tutorial-window-badge");
    const focusPoints = document.getElementById("tutorial-focus-points");
    const previewStage = document.getElementById("tutorial-preview-stage");
    const stepLabel = document.getElementById("tutorial-step-label");
    const progressFill = document.getElementById("tutorial-progress-fill");
    const stageChips = Array.from(document.querySelectorAll(".tutorial-stage-chip"));

    if (
        !shell ||
        !beginButton ||
        !nextButton ||
        !prevButton ||
        !exitTopButton ||
        !exitBottomButton ||
        !cardKicker ||
        !cardTitle ||
        !cardBody ||
        !windowBadge ||
        !focusPoints ||
        !previewStage ||
        !stepLabel ||
        !progressFill ||
        !stageChips.length
    ) {
        return;
    }

    const steps = [
        {
            kicker: "Home Screen",
            title: "Upload image from the home screen",
            body: "The farmer starts in Upload Images. This home window shows the upload area, preview panel, and upload button used to begin analysis.",
            badge: "Upload Window",
            focus: [
                "Upload Images is the first stop after login.",
                "The sample field photo is uploaded from the drop zone.",
                "The right panel confirms the image before sending it."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-dashboard">
                    <aside class="tutorial-mini-sidebar">
                        <span class="tutorial-mini-user">Farmer Dashboard</span>
                        <span class="tutorial-mini-nav is-active">Upload Images</span>
                        <span class="tutorial-mini-nav">View Results</span>
                        <span class="tutorial-mini-nav">Settings</span>
                        <span class="tutorial-mini-nav">Tutorial</span>
                    </aside>
                    <div class="tutorial-mini-main">
                        <div class="tutorial-mini-header">
                            <h4>Upload Image</h4>
                            <p>Start here to analyze your field photo.</p>
                        </div>
                        <div class="tutorial-mini-upload-layout">
                            <div class="tutorial-mini-dropzone tutorial-pulse-target">
                                <span class="tutorial-highlight-tag">Click here first</span>
                                <strong>Drop image here</strong>
                                <p>Use the sample corn field image</p>
                            </div>
                            <div class="tutorial-mini-preview-card">
                                <div class="tutorial-mini-image"></div>
                                <p>Preview panel</p>
                            </div>
                        </div>
                        <div class="tutorial-mini-upload-actions">
                            <span class="tutorial-mini-btn">Clear</span>
                            <span class="tutorial-mini-btn is-primary">Upload Image</span>
                        </div>
                    </div>
                </div>
            `
        },
        {
            kicker: "Image Selection",
            title: "Choose the sample image and confirm preview",
            body: "After the farmer clicks the drop zone, the sample image is selected and shown in the preview. The upload button is then used to send the image for analysis.",
            badge: "Upload Window",
            focus: [
                "The preview should match the actual uploaded field image.",
                "The file info confirms the right image was selected.",
                "Upload Image sends the photo into analysis."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-upload">
                    <div class="tutorial-mini-header">
                        <h4>Selected File Ready</h4>
                        <p>The farmer checks the photo before uploading.</p>
                    </div>
                    <div class="tutorial-upload-preview-grid">
                        <div class="tutorial-upload-file-card">
                            <p class="tutorial-file-name">corn-field-sample.jpg</p>
                            <p class="tutorial-file-size">2.84 MB</p>
                            <div class="tutorial-mini-status success">Image ready to upload</div>
                        </div>
                        <div class="tutorial-sample-photo tutorial-float-card">
                            <div class="tutorial-sample-photo-overlay">Uploaded image preview</div>
                        </div>
                    </div>
                    <div class="tutorial-mini-upload-actions">
                        <span class="tutorial-mini-btn">Clear</span>
                        <span class="tutorial-mini-btn is-primary tutorial-pulse-target">Upload Image</span>
                    </div>
                </div>
            `
        },
        {
            kicker: "Results Tab",
            title: "Open View Results to find finished analyses",
            body: "Once analysis is done, the farmer opens View Results. This window lists processed images, health bands, and shortcuts to the detailed result page or PDF report.",
            badge: "Results Window",
            focus: [
                "The newest result appears in the results gallery.",
                "Each card shows the health summary and band.",
                "Open Result leads to the full result page."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-results">
                    <div class="tutorial-mini-header">
                        <h4>View Results</h4>
                        <p>Open result cards and reports from here.</p>
                    </div>
                    <div class="tutorial-mini-filters">
                        <span>Search Result ID</span>
                        <span>Band Filter</span>
                        <span>Sort</span>
                    </div>
                    <div class="tutorial-result-card tutorial-float-card tutorial-pulse-target">
                        <div class="tutorial-result-thumb"></div>
                        <div class="tutorial-result-copy">
                            <h5>Result 240318-01</h5>
                            <p>Healthy center rows with stressed corners.</p>
                            <p>Band: WATCH</p>
                            <div class="tutorial-result-links">
                                <span>Open Result</span>
                                <span>PDF</span>
                            </div>
                        </div>
                    </div>
                </div>
            `
        },
        {
            kicker: "Result Window 1",
            title: "Crop and Segment Analysis window",
            body: "The first result window is the crop-and-segment workspace. The farmer adjusts the crop shape, runs cropped analysis, and checks the specific segment recommendation.",
            badge: "Crop & Segment",
            focus: [
                "Left side is the crop drawing area for selecting the farm region.",
                "The cropped heatzone preview appears on the right.",
                "Specific Segment Recommendation explains the selected segment."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-result-view">
                    <div class="tutorial-result-columns">
                        <div class="tutorial-pane tutorial-pane-large tutorial-pulse-target">
                            <h5>Crop & Segment Analysis</h5>
                            <div class="tutorial-canvas-frame">
                                <div class="tutorial-crop-overlay"></div>
                            </div>
                            <div class="tutorial-toolbar-row">
                                <span>Reset Shape</span>
                                <span>Clear Results</span>
                                <span class="is-primary">Analyze Cropped Area</span>
                            </div>
                        </div>
                        <div class="tutorial-pane">
                            <h5>Specific Segment Recommendation</h5>
                            <div class="tutorial-mini-heatzone"></div>
                            <p>Selected Segment: B3</p>
                            <p>Recommendation: Check water delivery on the lower edge.</p>
                        </div>
                    </div>
                </div>
            `
        },
        {
            kicker: "Result Window 2",
            title: "Recommendations and action board",
            body: "The next result window is the recommendation panel. This is where the farmer reads the action list, marks items done, and saves notes about work already performed in the field.",
            badge: "Recommendations",
            focus: [
                "Action cards turn the analysis into field tasks.",
                "Farmers can mark recommendations as done.",
                "Notes help track what was already applied."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-recommendations">
                    <div class="tutorial-pane tutorial-pane-full">
                        <h5>Results and Recommendation</h5>
                        <div class="tutorial-kpi-row">
                            <div class="tutorial-kpi-card"><span>Action Progress</span><strong>1 completed</strong></div>
                            <div class="tutorial-kpi-card"><span>Execution Style</span><strong>Field-ready task board</strong></div>
                        </div>
                        <div class="tutorial-rec-board tutorial-pulse-target">
                            <div class="tutorial-rec-item">
                                <span class="tutorial-priority">HIGH</span>
                                <p>Inspect stressed corner rows for water blockage.</p>
                            </div>
                            <div class="tutorial-rec-item">
                                <span class="tutorial-priority medium">MEDIUM</span>
                                <p>Review nutrient timing for the lower section.</p>
                            </div>
                        </div>
                        <div class="tutorial-notes-box">Farmer notes are saved here.</div>
                    </div>
                </div>
            `
        },
        {
            kicker: "Result Window 3",
            title: "Advanced Results and detailed values",
            body: "The final result window is Advanced Results. It contains useful RGB features, segment tables, management zone guidance, and deeper values for users who want more technical detail.",
            badge: "Advanced Results",
            focus: [
                "Advance Results opens the technical metrics section.",
                "Useful RGB Features summarize plant condition.",
                "Segment tables and management zones support deeper review."
            ],
            preview: `
                <div class="tutorial-mock tutorial-mock-advanced">
                    <div class="tutorial-pane tutorial-pane-full tutorial-pulse-target">
                        <div class="tutorial-advanced-header">
                            <h5>Useful RGB Features</h5>
                            <span class="tutorial-mini-btn is-primary">Advance Results</span>
                        </div>
                        <div class="tutorial-metric-grid">
                            <div class="tutorial-metric-card"><span>Green Coverage</span><strong>78%</strong></div>
                            <div class="tutorial-metric-card"><span>Stress Zone</span><strong>12%</strong></div>
                            <div class="tutorial-metric-card"><span>Vigor Score</span><strong>74/100</strong></div>
                            <div class="tutorial-metric-card"><span>Best Next Scan</span><strong>3 Days</strong></div>
                        </div>
                        <div class="tutorial-mini-table">
                            <div><span>Segment</span><span>Health</span><span>Vigor</span></div>
                            <div><span>A1</span><span>82</span><span>High</span></div>
                            <div><span>B3</span><span>48</span><span>Low</span></div>
                            <div><span>C2</span><span>67</span><span>Medium</span></div>
                        </div>
                    </div>
                </div>
            `
        }
    ];

    let currentStep = 0;

    function goHome() {
        if (window.dashboardNavigation?.setActiveWindow) {
            window.dashboardNavigation.setActiveWindow("upload");
            return;
        }
        document.getElementById("upload-button")?.click();
    }

    function renderFocusPoints(items) {
        focusPoints.innerHTML = "";
        items.forEach((item) => {
            const chip = document.createElement("div");
            chip.className = "tutorial-focus-chip";
            chip.textContent = item;
            focusPoints.appendChild(chip);
        });
    }

    function renderStep(stepIndex) {
        const step = steps[stepIndex];
        if (!step) {
            return;
        }

        currentStep = stepIndex;
        cardKicker.textContent = step.kicker;
        cardTitle.textContent = step.title;
        cardBody.textContent = step.body;
        windowBadge.textContent = step.badge;
        stepLabel.textContent = `Step ${stepIndex + 1} of ${steps.length}`;
        progressFill.style.width = `${((stepIndex + 1) / steps.length) * 100}%`;
        prevButton.disabled = stepIndex === 0;
        nextButton.textContent = stepIndex === steps.length - 1 ? "Finish Tutorial" : "Next";
        renderFocusPoints(step.focus);

        previewStage.classList.remove("is-entering");
        void previewStage.offsetWidth;
        previewStage.innerHTML = step.preview;
        previewStage.classList.add("is-entering");

        stageChips.forEach((chip, index) => {
            chip.classList.toggle("is-active", index === stepIndex);
        });
    }

    beginButton.addEventListener("click", () => renderStep(0));

    nextButton.addEventListener("click", () => {
        if (currentStep >= steps.length - 1) {
            goHome();
            return;
        }
        renderStep(currentStep + 1);
    });

    prevButton.addEventListener("click", () => {
        if (currentStep === 0) {
            return;
        }
        renderStep(currentStep - 1);
    });

    [exitTopButton, exitBottomButton].forEach((button) => {
        button.addEventListener("click", goHome);
    });

    stageChips.forEach((chip, index) => {
        chip.addEventListener("click", () => renderStep(index));
    });

    renderStep(0);
})();
