const searchInput = document.getElementById("result-search");
const bandFilter = document.getElementById("result-band-filter");
const sortSelect = document.getElementById("result-sort");
const gallery = document.querySelector("#results-window .results-gallery");

function getResultCards() {
    return Array.from(document.querySelectorAll("#results-window .result-card"));
}

function ensureNoResultsCard(message = "No results yet.") {
    if (!gallery) return;
    let emptyCard = gallery.querySelector(".no-results-state");
    if (!emptyCard) {
        emptyCard = document.createElement("div");
        emptyCard.className = "no-results-state";
        emptyCard.innerHTML = "<h3>No Results</h3><p></p>";
        gallery.appendChild(emptyCard);
    }
    const text = emptyCard.querySelector("p");
    if (text) {
        text.textContent = message;
    }
}

function applyResultsView() {
    const cards = getResultCards();
    if (!cards.length) {
        ensureNoResultsCard("No results yet. Upload and process an image to see entries.");
        return;
    }

    const q = (searchInput?.value || "").trim().toLowerCase();
    const band = (bandFilter?.value || "all").toLowerCase();
    const sortMode = sortSelect?.value || "newest";

    cards.forEach((card) => {
        const id = (card.dataset.resultId || "").toLowerCase();
        const itemBand = (card.dataset.healthBand || "").toLowerCase();

        const matchSearch = !q || id.includes(q);
        const matchBand = band === "all" || itemBand === band;
        card.style.display = matchSearch && matchBand ? "" : "none";
    });

    const parent = cards[0].parentElement;
    const visible = cards.filter((c) => c.style.display !== "none");
    const emptyCard = gallery?.querySelector(".no-results-state");

    visible.sort((a, b) => {
        const scoreA = Number(a.dataset.healthScore || 0);
        const scoreB = Number(b.dataset.healthScore || 0);
        const timeA = Date.parse(a.dataset.createdAt || "") || 0;
        const timeB = Date.parse(b.dataset.createdAt || "") || 0;
        if (sortMode === "oldest") return timeA - timeB;
        if (sortMode === "score_desc") return scoreB - scoreA;
        if (sortMode === "score_asc") return scoreA - scoreB;
        return timeB - timeA;
    });

    visible.forEach((card) => parent.appendChild(card));

    if (!visible.length) {
        ensureNoResultsCard("No matching results for the current filters.");
        if (emptyCard) {
            emptyCard.style.display = "block";
        }
    } else if (emptyCard) {
        emptyCard.style.display = "none";
    }
}

[searchInput, bandFilter, sortSelect].forEach((el) => {
    if (el) {
        el.addEventListener("input", applyResultsView);
        el.addEventListener("change", applyResultsView);
    }
});

gallery?.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (!target.classList.contains("result-delete-btn")) return;

    const resultId = target.dataset.resultId;
    if (!resultId) return;

    const confirmed = window.confirm(`Delete result ${resultId}? This also removes bucket files and cannot be undone.`);
    if (!confirmed) return;

    const originalText = target.textContent;
    target.textContent = "Deleting...";
    target.setAttribute("disabled", "true");

    try {
        const response = await fetch(`/api/results/${encodeURIComponent(resultId)}`, {
            method: "DELETE",
            headers: { Accept: "application/json" },
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            window.alert(payload.error || "Failed to delete result.");
            target.textContent = originalText || "Delete";
            target.removeAttribute("disabled");
            return;
        }

        const card = target.closest(".result-card");
        card?.remove();
        applyResultsView();
    } catch (_) {
        window.alert("Delete failed due to network or server error.");
        target.textContent = originalText || "Delete";
        target.removeAttribute("disabled");
    }
});

applyResultsView();
