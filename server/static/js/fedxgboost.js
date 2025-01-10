document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("training-parameters-form");

    form.addEventListener("submit", (event) => {
        const boostingRounds = parseInt(document.getElementById("boosting-rounds").value);
        const importanceRounds = parseInt(document.getElementById("importance-rounds").value);

        // Ensure that boosting_rounds is greater than or equal to importance_rounds
        if (boostingRounds < importanceRounds) {
            alert("Boosting Rounds must be greater than or equal to Importance Rounds.");
            event.preventDefault();
        }
    });
});
