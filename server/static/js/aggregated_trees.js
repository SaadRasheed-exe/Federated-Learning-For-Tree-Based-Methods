document.addEventListener("DOMContentLoaded", () => {
    const modelSelect = document.getElementById("model");
    const modelParametersContainer = document.getElementById("model-parameters-container");

    const modelParameterTemplates = {
        DecisionTreeClassifier: `
            <div class="form-group">
                <label for="max-depth">Max Depth:</label>
                <input type="number" id="max-depth" name="max_depth" min="1" value="5" required>
            </div>
        `,
        RandomForestClassifier: `
            <div class="form-group">
                <label for="n-estimators">Number of Estimators:</label>
                <input type="number" id="n-estimators" name="n_estimators" min="1" value="100" required>
            </div>
            <div class="form-group">
                <label for="max-depth">Max Depth:</label>
                <input type="number" id="max-depth" name="max_depth" min="1" value="10" required>
            </div>
        `,
        XGBClassifier: `
            <div class="form-group">
                <label for="n-estimators">Number of Estimators:</label>
                <input type="number" id="n-estimators" name="n_estimators" min="1" value="100" required>
            </div>
            <div class="form-group">
                <label for="max-depth">Max Depth:</label>
                <input type="number" id="max-depth" name="max_depth" min="1" value="6" required>
            </div>
            <div class="form-group">
                <label for="learning-rate">Learning Rate:</label>
                <input type="number" id="learning-rate" name="learning_rate" step="0.01" value="0.1" required>
            </div>
        `,
        LGBMClassifier: `
            <div class="form-group">
                <label for="n-estimators">Number of Estimators:</label>
                <input type="number" id="n-estimators" name="n_estimators" min="1" value="100" required>
            </div>
            <div class="form-group">
                <label for="max-depth">Max Depth:</label>
                <input type="number" id="max-depth" name="max_depth" min="1" value="6" required>
            </div>
            <div class="form-group">
                <label for="learning-rate">Learning Rate:</label>
                <input type="number" id="learning-rate" name="learning_rate" step="0.01" value="0.1" required>
            </div>
            <div class="form-group">
                <label for="boosting-type">Boosting Type:</label>
                <select id="boosting-type" name="boosting_type" required>
                    <option value="gbdt" selected>gbdt</option>
                    <option value="dart">dart</option>
                    <option value="rf">rf</option>
                </select>
            </div>
        `
    };
    
    
    

    modelSelect.addEventListener("change", () => {
        const selectedModel = modelSelect.value;

        if (selectedModel && modelParameterTemplates[selectedModel]) {
            modelParametersContainer.innerHTML = modelParameterTemplates[selectedModel];
            modelParametersContainer.classList.remove("hidden");
        } else {
            modelParametersContainer.innerHTML = "";
            modelParametersContainer.classList.add("hidden");
        }
    });

    const form = document.getElementById("training-parameters-form");
    const loadingSpinner = document.getElementById("loading-spinner");

    form.addEventListener("submit", function (event) {
        // Show loading spinner when the form is submitted
        loadingSpinner.style.visibility = 'visible';
    });
});


