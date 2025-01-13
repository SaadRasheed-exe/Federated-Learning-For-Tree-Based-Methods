document.addEventListener("DOMContentLoaded", () => {
    const methodButtons = document.querySelectorAll("#method-buttons button");
    const secondaryButtons = document.querySelectorAll("#disease-buttons button");
    const continueButton = document.getElementById("continue-btn");

    const methodInput = document.getElementById("method-input");
    const secondaryInput = document.getElementById("disease-input");

    let selectedMethodButton = null;
    let selectedDiseaseButton = null;

    // Function to handle button selection
    function handleSelection(buttons, selectedVar, inputField, callback) {
        buttons.forEach(button => {
            button.addEventListener("click", () => {
                // Clear selection for the group
                buttons.forEach(btn => btn.classList.remove("selected"));

                // Update the selected button
                if (selectedVar === button) {
                    selectedVar = null; // Deselect on second click
                    inputField.value = ""; // Clear input value
                } else {
                    button.classList.add("selected");
                    selectedVar = button;
                    inputField.value = button.textContent; // Set input value
                }

                // Callback to handle additional actions
                callback(selectedVar);
            });
        });
    }

    // Method buttons selection
    handleSelection(methodButtons, selectedMethodButton, methodInput, selected => {
        selectedMethodButton = selected;
        updateContinueButtonState();
    });

    // Disease buttons selection
    handleSelection(secondaryButtons, selectedDiseaseButton, secondaryInput, selected => {
        selectedDiseaseButton = selected;
        updateContinueButtonState();
    });

    const form = document.getElementById("selection-form");
    const loadingSpinner = document.getElementById("loading-spinner");

    form.addEventListener("submit", function (event) {
        // Show loading spinner when the form is submitted
        loadingSpinner.style.visibility = 'visible';
    });

    // Update Continue button state based on selections
    function updateContinueButtonState() {
        continueButton.disabled = !(selectedMethodButton && selectedDiseaseButton);
    }
});
