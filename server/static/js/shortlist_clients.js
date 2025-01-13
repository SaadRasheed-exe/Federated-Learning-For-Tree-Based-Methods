document.addEventListener("DOMContentLoaded", () => {
    const selectAllCheckbox = document.getElementById("select-all");
    const clientCards = document.querySelectorAll(".client-card");
    const continueButton = document.getElementById("continue-btn");
    let selectedClients = [];

    // Initially select all clients (based on "selected" class in HTML)
    clientCards.forEach(card => {
        card.classList.add("selected");
        const state = card.getAttribute("data-state");
        selectedClients.push(state);
    });

    // Enable Continue button if there are selected clients
    continueButton.disabled = selectedClients.length === 0;

    // Handle client card selection
    clientCards.forEach(card => {
        card.addEventListener("click", () => {
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

    // Toggle all selections based on the "Select All" checkbox
    selectAllCheckbox.addEventListener("change", () => {
        if (selectAllCheckbox.checked) {
            clientCards.forEach(card => {
                card.classList.add("selected");
                const state = card.getAttribute("data-state");
                if (!selectedClients.includes(state)) {
                    selectedClients.push(state);
                }
            });
        } else {
            clientCards.forEach(card => {
                card.classList.remove("selected");
                const state = card.getAttribute("data-state");
                selectedClients = selectedClients.filter(client => client !== state);
            });
        }
        // Enable or disable the continue button based on selections
        continueButton.disabled = selectedClients.length === 0;
    });

    // Continue button click handler
    continueButton.addEventListener("click", () => {
        if (selectedClients.length > 0) {
            const stateQueryParam = selectedClients.join(',');
            window.location.href = `/training-parameters?states=${stateQueryParam}`;
        }
    });
});
