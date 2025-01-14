from flask import Blueprint, jsonify, request
from ..CyclicXGBC import CyclicXGBClient
from xgboost import XGBClassifier

cyclic_blueprint = Blueprint('cyclic', __name__)
client = CyclicXGBClient()


@cyclic_blueprint.route('/train', methods=['POST'])
def train():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    model = data['model']
    num_rounds = data['num_rounds']
    updated_model = client.train(model, num_rounds)
    serialized = client.serialize_message({'model': updated_model})
    return jsonify({'serialized': serialized.hex()})

@cyclic_blueprint.route('/evaluate', methods=['POST'])
def evaluate():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    model = data['model']
    tp, fp, fn, tn = client.evaluate(model)
    return jsonify({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
