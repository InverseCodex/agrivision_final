const navButtons = document.querySelectorAll(".navbar-button");
const contentWindows = document.querySelectorAll(".content-window");
const mobileNavToggle = document.getElementById("mobile-nav-toggle");
const navbarOverlay = document.getElementById("navbar-overlay");
const mobileBreakpoint = 700;

function setActiveWindowFromButton(button) {
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
    if (window.innerWidth < mobileBreakpoint) {
        document.body.classList.remove("nav-open");
        if (mobileNavToggle) {
            mobileNavToggle.setAttribute("aria-expanded", "false");
        }
    }
}

function toggleMobileNav() {
    const isOpen = document.body.classList.toggle("nav-open");
    if (mobileNavToggle) {
        mobileNavToggle.setAttribute("aria-expanded", String(isOpen));
    }
}

navButtons.forEach((button) => {
    button.addEventListener("click", () => {
        setActiveWindowFromButton(button);
        closeMobileNav();
    });
});

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
    setActiveWindowFromButton(initialActiveButton);
}

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

