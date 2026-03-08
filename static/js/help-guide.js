(() => {
    const helpTour = document.getElementById("help-tour");
    const stepLabel = document.getElementById("tour-step-label");
    const stepTitle = document.getElementById("tour-step-title");
    const stepBody = document.getElementById("tour-step-body");
    const stepNext = document.getElementById("tour-next");
    const stepPrev = document.getElementById("tour-prev");
    const progress = document.getElementById("tour-progress");
    const jumpButtons = document.querySelectorAll(".help-jump-btn");
    const navButtons = document.querySelectorAll(".navbar-button");
    const contentWindows = document.querySelectorAll(".content-window");
    const mobileNavToggle = document.getElementById("mobile-nav-toggle");

    if (!helpTour || !stepLabel || !stepTitle || !stepBody || !stepNext || !stepPrev || !progress) {
        console.warn("Help guide elements not found.");
    } else {
        const steps = [
            {
                title: "Upload your first image",
                body: "Open Upload Images, pick your farm image, and click Upload Image.",
                target: "upload"
            },
            {
                title: "Check results",
                body: "Go to View Results to see your analysis card, score, and heatzone preview.",
                target: "results"
            },
            {
                title: "Download report",
                body: "Open a result and use the PDF button if you want a copy to share or print.",
                target: "results"
            },
            {
                title: "Personalize your dashboard",
                body: "Use Settings to choose your color style and update your display name.",
                target: "settings"
            }
        ];

        let currentStep = 0;

        function renderProgress() {
            progress.innerHTML = "";
            steps.forEach((_, index) => {
                const dot = document.createElement("span");
                dot.className = "tour-dot";
                if (index === currentStep) {
                    dot.classList.add("active");
                }
                progress.appendChild(dot);
            });
        }

        function updateTour() {
            const step = steps[currentStep];
            stepLabel.textContent = `Step ${currentStep + 1} of ${steps.length}`;
            stepTitle.textContent = step.title;
            stepBody.textContent = step.body;
            stepPrev.disabled = currentStep === 0;
            stepNext.textContent = currentStep === steps.length - 1 ? "Start Again" : "Next";

            const tourJumpButton = helpTour.querySelector(".help-jump-btn[data-target-window]");
            if (tourJumpButton) {
                tourJumpButton.dataset.targetWindow = step.target;
            }

            renderProgress();
        }

        function openWindow(windowName) {
            const navButton = document.getElementById(`${windowName}-button`);
            const windowPanel = document.getElementById(`${windowName}-window`);
            if (!navButton) {
                return;
            }

            navButtons.forEach((button) => button.classList.remove("button-active"));
            contentWindows.forEach((windowNode) => windowNode.classList.remove("window-active"));

            navButton.classList.add("button-active");
            if (windowPanel) {
                windowPanel.classList.add("window-active");
            }

            if (window.innerWidth < 700) {
                document.body.classList.remove("nav-open");
                if (mobileNavToggle) {
                    mobileNavToggle.setAttribute("aria-expanded", "false");
                }
            }
        }

        stepNext.addEventListener("click", () => {
            if (currentStep >= steps.length - 1) {
                currentStep = 0;
            } else {
                currentStep += 1;
            }
            updateTour();
        });

        stepPrev.addEventListener("click", () => {
            if (currentStep > 0) {
                currentStep -= 1;
                updateTour();
            }
        });

        jumpButtons.forEach((button) => {
            button.addEventListener("click", () => {
                const targetWindow = button.dataset.targetWindow;
                if (targetWindow) {
                    openWindow(targetWindow);
                }
            });
        });

        updateTour();
    }
})();
