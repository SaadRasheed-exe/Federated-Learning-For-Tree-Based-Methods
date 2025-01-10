document.addEventListener("DOMContentLoaded", () => {
    const aggregationTypeSelect = document.getElementById("aggregation-type");
    const aggregationParametersContainer = document.getElementById("aggregation-specific-parameters");

    const aggregationParameterTemplates = {
        Weighted: `
            <div class="form-group">
                <label for="total-boosting-rounds">Total Boosting Rounds:</label>
                <input type="number" id="total-boosting-rounds" name="total_boosting_rounds" min="1" value="100" required>
            </div>
        `,
        Unweighted: `
            <div class="form-group">
                <label for="boosting-rounds-per-node">Boosting Rounds per Node:</label>
                <input type="number" id="boosting-rounds-per-node" name="boosting_rounds_per_node" min="1" value="10" required>
            </div>
        `
    };

    aggregationTypeSelect.addEventListener("change", () => {
        const selectedAggregationType = aggregationTypeSelect.value;

        if (selectedAggregationType && aggregationParameterTemplates[selectedAggregationType]) {
            aggregationParametersContainer.innerHTML = aggregationParameterTemplates[selectedAggregationType];
            aggregationParametersContainer.classList.remove("hidden");
        } else {
            aggregationParametersContainer.innerHTML = "";
            aggregationParametersContainer.classList.add("hidden");
        }
    });
});
