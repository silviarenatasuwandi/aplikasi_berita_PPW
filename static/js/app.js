document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("classify-form");
    const urlInput = document.getElementById("url-input");
    const btnSubmit = document.getElementById("btn-submit");
    const btnText = btnSubmit.querySelector(".btn-text");
    const btnSpinner = btnSubmit.querySelector(".btn-spinner");

    const loadingCard = document.getElementById("loading-card");
    const errorCard = document.getElementById("error-card");
    const resultCard = document.getElementById("result-card");
    const errorMessage = document.getElementById("error-message");

    const presetBtns = document.querySelectorAll(".preset-btn");

    // Handle Preset URL Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const sampleUrl = btn.getAttribute("data-url");
            if (sampleUrl) {
                urlInput.value = sampleUrl;
                form.dispatchEvent(new Event("submit", { cancelable: true }));
            }
        });
    });

    // Form Submission Event
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const url = urlInput.value.trim();

        if (!url) return;

        // UI Reset
        hideAllCards();
        setLoadingState(true);
        simulateSteps();

        try {
            const response = await fetch("/api/classify", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ url: url })
            });

            const result = await response.json();

            setLoadingState(false);

            if (result.status === "success") {
                renderResult(result.data);
            } else {
                showError(result.message || "Terjadi kesalahan saat memproses URL.");
            }
        } catch (err) {
            setLoadingState(false);
            showError("Tidak dapat terhubung ke server backend Flask. Silakan coba lagi.");
        }
    });

    function hideAllCards() {
        loadingCard.classList.add("hidden");
        errorCard.classList.add("hidden");
        resultCard.classList.add("hidden");
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            btnSubmit.disabled = true;
            btnText.classList.add("hidden");
            btnSpinner.classList.remove("hidden");
            loadingCard.classList.remove("hidden");
        } else {
            btnSubmit.disabled = false;
            btnText.classList.remove("hidden");
            btnSpinner.classList.add("hidden");
            loadingCard.classList.add("hidden");
        }
    }

    function simulateSteps() {
        const steps = ["step-1", "step-2", "step-3", "step-4"];
        steps.forEach(s => document.getElementById(s).classList.remove("active"));
        
        document.getElementById("step-1").classList.add("active");

        setTimeout(() => {
            document.getElementById("step-2")?.classList.add("active");
        }, 700);

        setTimeout(() => {
            document.getElementById("step-3")?.classList.add("active");
        }, 1400);

        setTimeout(() => {
            document.getElementById("step-4")?.classList.add("active");
        }, 2100);
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorCard.classList.remove("hidden");
    }

    function renderResult(data) {
        const label = (data.label || "other").toLowerCase();
        
        // Element references
        const badge = document.getElementById("result-badge");
        const badgeIcon = document.getElementById("badge-icon");
        const badgeText = document.getElementById("badge-text");

        // Set Badge Styling
        badge.className = "result-badge";
        if (label === "sport") {
            badge.classList.add("badge-sport");
            badgeIcon.className = "fa-solid fa-trophy";
            badgeText.textContent = "SPORT";
        } else if (label === "finance") {
            badge.classList.add("badge-finance");
            badgeIcon.className = "fa-solid fa-sack-dollar";
            badgeText.textContent = "FINANCE";
        } else {
            badge.classList.add("badge-other");
            badgeIcon.className = "fa-solid fa-layer-group";
            badgeText.textContent = "OTHER";
        }

        // Confidence
        document.getElementById("confidence-value").textContent = `${data.confidence}%`;
        document.getElementById("explanation-text").textContent = data.explanation;

        // Article Meta
        document.getElementById("article-title").innerHTML = `<i class="fa-solid fa-newspaper"></i> ${escapeHtml(data.title)}`;
        const linkElem = document.getElementById("article-url-link");
        linkElem.href = data.url;
        linkElem.textContent = data.url;

        // Probabilities
        const probs = data.probabilities || {};
        const sportPct = probs.sport || 0;
        const financePct = probs.finance || 0;
        const otherPct = probs.other || 0;

        document.getElementById("prob-sport-val").textContent = `${sportPct}%`;
        document.getElementById("bar-sport").style.width = `${sportPct}%`;

        document.getElementById("prob-finance-val").textContent = `${financePct}%`;
        document.getElementById("bar-finance").style.width = `${financePct}%`;

        document.getElementById("prob-other-val").textContent = `${otherPct}%`;
        document.getElementById("bar-other").style.width = `${otherPct}%`;

        // Accordion Details
        document.getElementById("full-article-text").textContent = data.full_article;
        document.getElementById("token-count").textContent = data.token_count;

        // Token cloud tags
        const tokensCloud = document.getElementById("tokens-cloud");
        tokensCloud.innerHTML = "";
        if (data.tokens && data.tokens.length > 0) {
            data.tokens.forEach(token => {
                const tag = document.createElement("span");
                tag.className = "token-tag";
                tag.textContent = token;
                tokensCloud.appendChild(tag);
            });
        }

        resultCard.classList.remove("hidden");
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
});
