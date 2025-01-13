from flask import Flask, render_template, request
from core import FedXGBServer, AggregatedTreesServer, CyclicXGBServer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
server = None
disease = None
method = None
selected_states = None
datastats = None

disease_map = {
    'Diabetes': 'Diabetes_E11',
    'Hypertension': 'Hypertension_I10'
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/shortlist-clients", methods=["POST"])
def shortlist_clients():
    global server, disease, method, datastats
    # Retrieve selected values from the form
    method = request.form.get("method")
    disease = request.form.get("disease")
    client_json_path = 'static/res/clients.json'

    if method == "Aggregated Trees":
        server = AggregatedTreesServer(client_json_path)
    elif method == "FedXGBoost":
        server = FedXGBServer(client_json_path)
    elif method == "Cyclic XGBoost":
        server = CyclicXGBServer(client_json_path)
    else:
        return "Invalid method selected", 400
    
    
    if not os.path.exists(f'static/res/{disease_map[disease]}'):
        raise FileNotFoundError(f"Directory not found: {disease_map[disease]}")
    
    server.send_code_dir(f'static/res/{disease_map[disease]}/config.ini')
    datastats = server.get_data_stats()

    # Pass selected values to the result page
    return render_template("shortlist_clients.html", datastats=datastats)

@app.route("/training-parameters")
def training_parameters():
    global selected_states, method
    states = request.args.get("states")
    selected_states = states.split(',') if states else []
        
    # Render the appropriate page based on the method
    if method == "Aggregated Trees":
        return render_template("aggregated_trees.html")
    elif method == "Cyclic XGBoost":
        return render_template("cyclic_xgboost.html")
    elif method == "FedXGBoost":
        return render_template("fedxgboost.html")
    else:
        return "Invalid method selected", 400

@app.route("/results", methods=["POST"])
def results():
    global datastats, selected_states, method
    
    scores_html = pd.DataFrame().to_html(index=True, classes='table table-striped table-bordered')
    testdata = pd.read_csv(f'static/res/{disease_map[disease]}/testdata.csv')
    X_test = testdata.drop(columns=['is_diagnosed'])
    y_test = testdata['is_diagnosed']
    agg_model = None

    if method == "Aggregated Trees":
        # Retrieve form data
        aggregation_type = request.form.get("aggregation_type")
        model = request.form.get("model")
        params = {key: request.form[key] for key in request.form if key not in ["aggregation_type", "model"]}

        weightage = None
        if aggregation_type == "Weighted":
            weightage = {state: (stat['diagnosed']['n_samples'] + stat['normal']['n_samples']) for state, stat in datastats.items() if state in selected_states}
        
        
        params = server.parse_params(params)
        model = eval(model)(**params)
        agg_model = server.fit(model, weightage, selected_states)
        scores = server.evaluate(agg_model)
    
    elif method == "FedXGBoost":
        params = {key: request.form[key] for key in request.form}
        params = server.parse_params(params)
        server.initialize(params['avg_splits'])
        agg_model = server.fit(
            resume=False,
            min_child_weight = params['min_child_weight'],
            depth = params['depth'],
            min_leaf = params['min_leaf'],
            learning_rate = params['learning_rate'],
            boosting_rounds = params['boosting_rounds'],
            lambda_ = params['lambda_'],
            gamma = params['gamma'],
            features_per_booster = params['features_per_booster'],
            importance_rounds = params['importance_rounds']
        )
        scores = server.evaluate()
        X_test = X_test[agg_model.feature_names]
    
    elif method == "Cyclic XGBoost":
        aggregation_type = request.form.get("aggregation_type")
        params = {key: request.form[key] for key in request.form if key not in ["aggregation_type", "total_boosting_rounds", "boosting_rounds_per_node"]}

        weightage = None
        if aggregation_type == "Weighted":
            total_trees = int(request.form.get("total_boosting_rounds"))
            weightage = {}
            total = sum([stat['diagnosed']['n_samples'] + stat['normal']['n_samples'] for state, stat in datastats.items() if state in selected_states])
            for state in selected_states:
                weightage[state] = int(total_trees * ((datastats[state]['diagnosed']['n_samples'] + datastats[state]['normal']['n_samples']) / total))
        
        elif aggregation_type == "Unweighted":
            num_rounds = int(request.form.get("boosting_rounds_per_node"))
            weightage = {state: num_rounds for state in selected_states}
        
        else:
            raise ValueError("Invalid aggregation type")

        params = server.parse_params(params)
        model = XGBClassifier(**params)

        agg_model = server.fit(model, weightage=weightage)
        scores = server.evaluate(agg_model)
        X_test = X_test[agg_model.feature_names_in_]

    
    scores_df = pd.DataFrame(scores, index=[0])
    scores_df.columns = [col.capitalize() for col in scores_df.columns]
    scores_df = scores_df.round(4)
    scores_df.index = ['Train']

    test_scores = {}
    if agg_model:
        y_pred = agg_model.predict(X_test)
        test_scores['Accuracy'] = accuracy_score(y_test, y_pred)
        test_scores['Precision'] = precision_score(y_test, y_pred)
        test_scores['Recall'] = recall_score(y_test, y_pred)
        test_scores['F1'] = f1_score(y_test, y_pred)
    
    scores_df.loc['Test', :] = test_scores
    scores_html = scores_df.to_html(index=True, classes='table table-striped table-bordered', float_format='%.2f')

    # Render the results page
    return render_template(
        "results.html",
        table=scores_html
    )

if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=7224,
        debug=True,
    )
