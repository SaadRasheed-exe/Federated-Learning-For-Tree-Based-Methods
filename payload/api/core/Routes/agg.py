from flask import Blueprint, jsonify, request
from ..AggTreesC import AggregatedTreesClient
from ..Models import MajorityVotingEnsemble
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


agg_blueprint = Blueprint('agg', __name__)
client = AggregatedTreesClient()


@agg_blueprint.route('/train', methods=['POST'])
def train():
    data = request.json['data']
    serialized_model = data['model']
    serialized_time = data['time']
    feature_names = data['feature_names']

    model = pickle.loads(bytes.fromhex(serialized_model))
    time = datetime.strptime(serialized_time, '%Y-%m-%d %H:%M:%S')
    
    trained_model = client.train(model, time, feature_names)
    serialized_model = pickle.dumps(trained_model).hex()
    return jsonify({'model': serialized_model})

@agg_blueprint.route('/get_feature_names', methods=['POST'])
def get_feature_names():
    feature_names = client.get_feature_names()
    return jsonify({'feature_names': feature_names})

@agg_blueprint.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json['data']
    serialized_model = data['model']

    model_data = pickle.loads(bytes.fromhex(serialized_model))
    model = MajorityVotingEnsemble(model_data['models'], model_data['model_weightage'])
    tp, fp, fn, tn = client.evaluate(model)

    return jsonify({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})