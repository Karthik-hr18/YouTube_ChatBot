document.addEventListener("DOMContentLoaded", () => {
    const askForm = document.getElementById("askForm");
    const searchButton = document.getElementById("searchButton");
    const videoIdInput = document.getElementById("videoIdInput");
    const searchInput = document.getElementById("searchInput");
    const resultsDiv = document.getElementById("results");

    askForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const videoId = videoIdInput.value.trim();
        const question = searchInput.value.trim();
        if (!videoId || !question) return;

        searchButton.disabled = true;
        resultsDiv.textContent = "Searching...";
        try {
            const response = await fetch("http://127.0.0.1:8000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_id: videoId,
                    question,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Request failed");
            }

            resultsDiv.replaceChildren();
            const answer = document.createElement("p");
            answer.textContent = data.answer;
            resultsDiv.append(answer);

            if (data.citations?.length) {
                const citationsHeading = document.createElement("h2");
                citationsHeading.textContent = "Sources";
                resultsDiv.append(citationsHeading);

                const citationsList = document.createElement("ul");
                for (const citation of data.citations) {
                    const item = document.createElement("li");
                    const link = document.createElement("a");
                    link.href = citation.url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = citation.timestamp;
                    item.append(link);
                    citationsList.append(item);
                }
                resultsDiv.append(citationsList);
            }
        } catch (error) {
            console.error("Error asking about transcript:", error);
            resultsDiv.textContent = error.message || "An error occurred.";
        } finally {
            searchButton.disabled = false;
        }
    });
});