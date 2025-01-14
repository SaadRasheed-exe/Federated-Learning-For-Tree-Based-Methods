async function updateProgress() {
    try {
        const response = await fetch('/progress');
        const data = await response.json();

        const progressBar = document.getElementById("progress-bar");
        const progressText = document.getElementById("progress-text");
        const percentage = Math.round((data.round / data.total) * 100);
        progressBar.style.width = percentage + "%";
        progressText.innerText = `Round ${data.round} of ${data.total}`;
        
        if (data.round < data.total) {
            setTimeout(updateProgress, 500); // Poll every 500ms
        } else {
            document.getElementById("completion-message").innerText = "Training Complete!";
            // Wait for 1 second before redirecting to the results page
            await new Promise(resolve => setTimeout(resolve, 1000));
            window.location.href = "/results";
        }
    } catch (error) {
        console.error("Error updating progress:", error);
    }
}

document.addEventListener("DOMContentLoaded", updateProgress);