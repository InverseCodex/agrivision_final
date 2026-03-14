(() => {
    const mobileBreakpoint = 700;
    let sharedNavButtons = [];
    let sharedContentWindows = [];

    function dispatchWindowChange(button) {
        if (!button) return;
        const windowName = button.id.replace("-button", "");
        document.dispatchEvent(
            new CustomEvent("dashboard:windowchange", {
                detail: { windowName }
            })
        );
    }

    function setActiveWindowFromButton(button, navButtons, contentWindows) {
        const navUrl = button.dataset.navUrl;
        if (navUrl) {
            window.location.href = navUrl;
            return;
        }

        const buttonId = button.id;
        const windowId = buttonId.replace("-button", "-window");
        const targetWindow = document.getElementById(windowId);

        navButtons.forEach((btn) => btn.classList.remove("button-active"));
        contentWindows.forEach((win) => win.classList.remove("window-active"));

        button.classList.add("button-active");
        if (targetWindow) {
            targetWindow.classList.add("window-active");
        }
        dispatchWindowChange(button);
    }

    function setActiveWindowByName(windowName) {
        if (!windowName || !sharedNavButtons.length || !sharedContentWindows.length) {
            return;
        }

        const targetButton = document.getElementById(`${windowName}-button`);
        if (!targetButton) {
            return;
        }

        setActiveWindowFromButton(targetButton, sharedNavButtons, sharedContentWindows);
        closeMobileNav();
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
        const navButtons = Array.from(document.querySelectorAll(".navbar-button"));
        const contentWindows = Array.from(document.querySelectorAll(".content-window"));
        const mobileNavToggle = document.getElementById("mobile-nav-toggle");
        const navbarOverlay = document.getElementById("navbar-overlay");
        if (!navButtons.length || !contentWindows.length) {
            return;
        }

        sharedNavButtons = navButtons;
        sharedContentWindows = contentWindows;
        window.dashboardNavigation = {
            setActiveWindow: setActiveWindowByName,
            closeMobileNav
        };

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

