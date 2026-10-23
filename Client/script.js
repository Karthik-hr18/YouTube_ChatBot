document.addEventListener("DOMContentLoaded", () => {
    const askForm = document.getElementById("askForm");
    const searchButton = document.getElementById("searchButton");
    const videoUrlInput = document.getElementById("videoUrlInput");
    const questionInput = document.getElementById("questionInput");
    const refreshInput = document.getElementById("refreshInput");
    const resultsSection = document.getElementById("results");
    const apiBaseUrl = (
        window.YOUTUBE_CHAT_CONFIG?.apiBaseUrl || "http://127.0.0.1:8000"
    ).replace(/\/+$/, "");

    async function readJson(response) {
        try {
            return await response.json();
        } catch {
            throw new Error(`The server returned an unreadable response (${response.status}).`);
        }
    }

    function showMessage(message, kind = "status") {
        resultsSection.replaceChildren();
        resultsSection.className = `results ${kind}`;
        const paragraph = document.createElement("p");
        paragraph.textContent = message;
        resultsSection.append(paragraph);
    }

    async function waitForJob(statusUrl) {
        const startedAt = Date.now();
        const maximumWaitMs = 30 * 60 * 1000;

        while (Date.now() - startedAt < maximumWaitMs) {
            await new Promise((resolve) => window.setTimeout(resolve, 1200));
            const response = await fetch(statusUrl);
            const job = await readJson(response);

            if (!response.ok) {
                throw new Error(job.detail || "Could not check the answer status.");
            }

            if (job.status === "completed") return job.result;
            if (job.status === "failed") throw new Error(job.error);

            showMessage(
                job.status === "processing"
                    ? "Working on the transcript and preparing your answer…"
                    : "Your request is queued…"
            );
        }

        throw new Error("This is taking longer than expected. Please try again shortly.");
    }

    function displayResult(result) {
        resultsSection.replaceChildren();
        resultsSection.className = "results answer";

        const answerHeading = document.createElement("h2");
        answerHeading.textContent = "Answer";
        const answer = document.createElement("p");
        answer.className = "answer-text";
        answer.textContent = result.answer;
        resultsSection.append(answerHeading, answer);

        if (!result.citations?.length) return;

        const sourcesHeading = document.createElement("h3");
        sourcesHeading.textContent = "Transcript sources";
        const citationsList = document.createElement("ul");
        citationsList.className = "citation-list";

        for (const citation of result.citations) {
            const item = document.createElement("li");
            const link = document.createElement("a");
            link.href = citation.url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = citation.timestamp_end
                && citation.timestamp_end !== citation.timestamp
                ? `${citation.timestamp}–${citation.timestamp_end}`
                : citation.timestamp;
            item.append(link);
            citationsList.append(item);
        }

        resultsSection.append(sourcesHeading, citationsList);
    }

    askForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const videoUrl = videoUrlInput.value.trim();
        const question = questionInput.value.trim();
        if (!videoUrl || !question) return;

        searchButton.disabled = true;
        resultsSection.setAttribute("aria-busy", "true");
        showMessage("Sending your question…");

        try {
            const response = await fetch(`${apiBaseUrl}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_url: videoUrl,
                    question,
                    refresh: refreshInput.checked,
                }),
            });
            const job = await readJson(response);

            if (!response.ok) {
                throw new Error(job.detail || "Could not submit your question.");
            }

            const statusUrl = new URL(job.status_url, apiBaseUrl);
            const result = await waitForJob(statusUrl);
            displayResult(result);
        } catch (error) {
            console.error("Error asking about video:", error);
            showMessage(error.message || "Something went wrong. Please try again.", "error");
        } finally {
            searchButton.disabled = false;
            resultsSection.setAttribute("aria-busy", "false");
        }
    });
});
