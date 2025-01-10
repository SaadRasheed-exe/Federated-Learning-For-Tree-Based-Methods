document.addEventListener("DOMContentLoaded", () => {
    const clientCards = document.querySelectorAll(".client-card");
    const continueButton = document.getElementById("continue-btn");
    let selectedClients = [];

    // Handle client card selection
    clientCards.forEach(card => {
        card.addEventListener("click", () => {
            // Toggle selection for each card independently
            const state = card.getAttribute("data-state");
            if (card.classList.contains("selected")) {
                // Deselect the card
                card.classList.remove("selected");
                selectedClients = selectedClients.filter(client => client !== state);
            } else {
                // Select the card
                card.classList.add("selected");
                selectedClients.push(state);
            }

            // Enable or disable the continue button based on selections
            continueButton.disabled = selectedClients.length === 0;
        });
    });

    // Continue button click handler
    continueButton.addEventListener("click", () => {
        if (selectedClients.length > 0) {
            // Redirect to the Training Parameters page with the selected states and method
            const stateQueryParam = selectedClients.join(',');
            window.location.href = `/training-parameters?states=${stateQueryParam}`;
        }
    });
});
