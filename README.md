# Introduction 
This is an implementation of a tree ensemble-based federated learning framework for disease prognosis. There are three methods here:

1. **Aggregated Trees**
2. **Cyclic XGBoost**
3. **FedXGBoost**

This was built using **Python 3.10**.

- The **`payload`** folder contains the code files that are served to the nodes via Flask URLs.
- The **`server`** folder contains the files for the central aggregator, it handles client communication and the overall federated training procedure.

# Getting Started
1. Set up a **virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:

    - For **macOS/Linux**:

      ```bash
      source venv/bin/activate
      ```

    - For **Windows**:

      ```bash
      .\venv\Scripts\activate
      ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask app located in the `server` folder:

    ```bash
    python server/app.py
    ```

# Build and Test
1. Open the Flask app in your browser.
2. Select the model, available states, and configure the model parameters.
3. The model will train, and the results will load.

**Note:** The test datasets have not been uploaded, so test results will not be displayed unless the datasets are provided.

- The **`best_params.json`** file contains a set of parameters that are efficient for the algorithm and give good results without excessive computation time.
- The **centralized results** are stored in `server/static/res/Central_results`.

# Contribute
Contribute by:

- Optimizing **FedXGBoost** or proposing new methods for federated learning with tree ensembles.
- Suggesting improvements to the server-side code or the user interface.
- Proposing new features or enhancements.

Please fork the repository, create a feature branch, and submit a pull request.
