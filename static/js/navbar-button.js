(() => {
    const mobileBreakpoint = 700;

    function setActiveWindowFromButton(button, navButtons, contentWindows) {
        const buttonId = button.id;
        const windowId = buttonId.replace("-button", "-window");
        const targetWindow = document.getElementById(windowId);

        navButtons.forEach((btn) => btn.classList.remove("button-active"));
        contentWindows.forEach((win) => win.classList.remove("window-active"));

        button.classList.add("button-active");
        if (targetWindow) {
            targetWindow.classList.add("window-active");
        }
    }

    function closeMobileNav() {
        const mobileNavToggle = document.getElementById("mobile-nav-toggle");
        if (window.innerWidth < mobileBreakpoint) {
            document.body.classList.remove("nav-open");
            if (mobileNavToggle) {
                mobileNavToggle.setAttribute("aria-expanded", "false");
            }
        }
    }

    function toggleMobileNav() {
        const mobileNavToggle = document.getElementById("mobile-nav-toggle");
        const isOpen = document.body.classList.toggle("nav-open");
        if (mobileNavToggle) {
            mobileNavToggle.setAttribute("aria-expanded", String(isOpen));
        }
    }

    function initializeNavbarButtons() {
        const navButtons = document.querySelectorAll(".navbar-button");
        const contentWindows = document.querySelectorAll(".content-window");
        const mobileNavToggle = document.getElementById("mobile-nav-toggle");
        const navbarOverlay = document.getElementById("navbar-overlay");
        if (!navButtons.length || !contentWindows.length) {
            return;
        }

        const initialActiveButton =
            (function resolveInitialButton() {
                const requestedWindow = new URLSearchParams(window.location.search).get("window");
                if (requestedWindow) {
                    const requestedButton = document.getElementById(`${requestedWindow}-button`);
                    if (requestedButton) return requestedButton;
                }
                return document.querySelector(".navbar-button.button-active") || navButtons[0];
            })();
        if (initialActiveButton) {
            setActiveWindowFromButton(initialActiveButton, navButtons, contentWindows);
        }

        document.addEventListener("click", (event) => {
            const button = event.target.closest(".navbar-button");
            if (!button) return;
            setActiveWindowFromButton(button, navButtons, contentWindows);
            closeMobileNav();
        });

        if (mobileNavToggle) {
            mobileNavToggle.addEventListener("click", toggleMobileNav);
        }

        if (navbarOverlay) {
            navbarOverlay.addEventListener("click", closeMobileNav);
        }

        window.addEventListener("resize", () => {
            if (window.innerWidth >= mobileBreakpoint) {
                document.body.classList.remove("nav-open");
                if (mobileNavToggle) {
                    mobileNavToggle.setAttribute("aria-expanded", "false");
                }
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeNavbarButtons);
    } else {
        initializeNavbarButtons();
    }
})();

