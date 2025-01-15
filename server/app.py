from flask import Flask, render_template, request, jsonify, redirect
from core import FedXGBServer, AggregatedTreesServer, CyclicXGBServer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import threading
import shap
import warnings
import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


app = Flask(__name__)
server = None
disease = None
method = None
selected_states = None
datastats = None
scores = None
agg_model = None
progress = None
X_test = None
y_test = None

DISEASE_MAPPING = {
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
        if server is None or server.__class__.__name__ != "FedXGBServer":
            server = FedXGBServer(client_json_path)
    elif method == "Cyclic XGBoost":
        server = CyclicXGBServer(client_json_path)
    else:
        return "Invalid method selected", 400
    
    if not os.path.exists(f'static/res/{DISEASE_MAPPING[disease]}'):
        raise FileNotFoundError(f"Directory not found: {DISEASE_MAPPING[disease]}")
    
    server.send_code_dir(f'static/res/{DISEASE_MAPPING[disease]}/config.ini')
    datastats = server.get_data_stats()

    # Pass selected values to the result page
    return render_template("shortlist_clients.html", datastats=datastats)

@app.route("/training-parameters")
def training_parameters():
    global selected_states, method
    states = request.args.get("states")
    selected_states = states.split(',') if states else []
    server.client_manager.active_clients = selected_states
        
    # Render the appropriate page based on the method
    if method == "Aggregated Trees":
        return render_template("aggregated_trees.html")
    elif method == "Cyclic XGBoost":
        return render_template("cyclic_xgboost.html")
    elif method == "FedXGBoost":
        return render_template("fedxgboost.html")
    else:
        return "Invalid method selected", 400
    

@app.route("/progress", methods=["GET"])
def get_progress():
    return jsonify(progress)

def train_with_progress(generator, total_rounds):
    global progress, agg_model
    progress["total"] = total_rounds  # Assume total rounds are known
    try:
        while True:
            progress["round"] = next(generator)
            if progress["interrupted"]:
                break
    except StopIteration as e:
        progress["round"] = progress["total"]  # Ensure completion
        progress["completed"] = True
        agg_model = e.value
        return e.value
    
@app.route("/interrupt-training", methods=["POST"])
def interrupt_training():
    global progress
    progress["interrupted"] = True
    return redirect("/")

@app.route("/training", methods=["POST"])
def training():
    global datastats, selected_states, method, agg_model, progress
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
    
    elif method == "FedXGBoost":
        params = {key: request.form[key] for key in request.form}
        params = server.parse_params(params)
        server.initialize(params['avg_splits'])
        generator = server.fit_generator(
            resume=False,
            min_child_weight=params['min_child_weight'],
            depth=params['depth'],
            min_leaf=params['min_leaf'],
            learning_rate=params['learning_rate'],
            boosting_rounds=params['boosting_rounds'],
            lambda_=params['lambda_'],
            gamma=params['gamma'],
            features_per_booster=params['features_per_booster'],
            importance_rounds=params['importance_rounds']
        )

        progress = {"round": 0, "total": params["boosting_rounds"], "completed": False, "interrupted": False}
        thread = threading.Thread(target=train_with_progress, args=(generator, params['boosting_rounds']))
        thread.start()

        return render_template("training_progress.html")
    
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

    return redirect("/results")

@app.route("/results", methods=["GET", "POST"])
def results():
    global datastats, selected_states, method, train_scores, agg_model, progress, X_test, y_test

    testdata = pd.read_csv(f'static/res/{DISEASE_MAPPING[disease]}/testdata.csv')
    X_test = testdata.drop(columns=['is_diagnosed'])
    y_test = testdata['is_diagnosed']

    train_scores = server.evaluate(agg_model)

    X_test = X_test[agg_model.feature_names_in_]
    scores_df = pd.DataFrame(train_scores, index=[0])
    scores_df.columns = [col.capitalize() for col in scores_df.columns]
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
    progress = None

    return render_template(
        "results.html",
        table=scores_html
    )


@app.route("/explain", methods=["GET"])
def explain():
    global agg_model, X_test, method

    X_sample = X_test.sample(100, random_state=42)

    plot_path = "static/feature_importance.png"

    if method == "Aggregated Trees":
        shap_values = agg_model.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=12)  # Set show=False to prevent automatic display
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    elif method == "Cyclic XGBoost":
        explainer = shap.TreeExplainer(agg_model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    elif method == "FedXGBoost":
        features = agg_model.feature_names_in_
        feature_importance = agg_model.feature_importance

        # Ensure all features are accounted for in feature importance
        feature_importance = {
            features[i]: feature_importance.get(i, 0) for i in range(len(features))
        }

        # Sort the features by importance
        sorted_feature_importance = {
            k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        }

        # Prepare data for plotting
        sorted_features = list(sorted_feature_importance.keys())[:12] # Top 12 features
        sorted_values = list(sorted_feature_importance.values())[:12]

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_values, color='skyblue')
        plt.xlabel('Feature Importance (Gain)')
        plt.ylabel('Features')
        plt.title('Feature Importance Plot')
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    else:
        raise ValueError("Invalid method")

    return render_template("explain.html", method=method)


if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=7224,
        debug=True,
    )
