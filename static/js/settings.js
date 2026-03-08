const SETTINGS_KEYS = {
    theme: "site_theme",
    motionEnabled: "site_motion_enabled"
};

const DEFAULTS = {
    theme: "default",
    motionEnabled: true
};

function getSavedTheme() {
    return localStorage.getItem(SETTINGS_KEYS.theme) || DEFAULTS.theme;
}

function getSavedMotionEnabled() {
    const saved = localStorage.getItem(SETTINGS_KEYS.motionEnabled);
    if (saved === null) {
        return DEFAULTS.motionEnabled;
    }
    return saved === "true";
}

function applyTheme(theme) {
    const safeTheme = ["default", "neon", "minimalist"].includes(theme) ? theme : DEFAULTS.theme;
    document.body.setAttribute("data-theme", safeTheme);
    localStorage.setItem(SETTINGS_KEYS.theme, safeTheme);

    document.querySelectorAll(".theme-option").forEach((button) => {
        const isActive = button.dataset.themeValue === safeTheme;
        button.classList.toggle("theme-option-active", isActive);
    });
}

function applyUsername(username) {
    const clean = (username || "").trim() || "USER";

    const userNameElement = document.querySelector(".user-name");
    if (userNameElement) {
        userNameElement.textContent = clean.toUpperCase();
    }

    const usernameInput = document.getElementById("username-input");
    if (usernameInput) {
        usernameInput.value = clean;
    }
}

function applyMotionSetting(enabled) {
    localStorage.setItem(SETTINGS_KEYS.motionEnabled, String(enabled));
    document.body.classList.toggle("reduce-motion", !enabled);

    const motionToggle = document.getElementById("motion-toggle");
    if (motionToggle) {
        motionToggle.checked = enabled;
    }
}

function resetSettings() {
    applyTheme(DEFAULTS.theme);
    applyMotionSetting(DEFAULTS.motionEnabled);
}

function initializeSettings() {
    applyTheme(getSavedTheme());
    const initialName = document.querySelector(".user-name")?.textContent || "USER";
    applyUsername(initialName);
    applyMotionSetting(getSavedMotionEnabled());

    document.querySelectorAll(".theme-option").forEach((button) => {
        button.addEventListener("click", () => {
            applyTheme(button.dataset.themeValue);
        });
    });

    const usernameForm = document.getElementById("username-form");
    if (usernameForm) {
        usernameForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const usernameInput = document.getElementById("username-input");
            if (usernameInput) {
                const newUsername = usernameInput.value.trim();
                if (!newUsername) {
                    return;
                }

                try {
                    const response = await fetch("/account/username", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ username: newUsername })
                    });
                    const payload = await response.json();
                    if (!response.ok) {
                        alert(payload.error || "Unable to update username.");
                        return;
                    }
                    applyUsername(payload.username || newUsername);
                } catch (error) {
                    alert("Unable to update username right now.");
                }
            }
        });
    }

    const motionToggle = document.getElementById("motion-toggle");
    if (motionToggle) {
        motionToggle.addEventListener("change", (event) => {
            applyMotionSetting(Boolean(event.target.checked));
        });
    }

    const resetButton = document.getElementById("reset-settings");
    if (resetButton) {
        resetButton.addEventListener("click", resetSettings);
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeSettings);
} else {
    initializeSettings();
}
