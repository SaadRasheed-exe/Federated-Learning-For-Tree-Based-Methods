from flask import Blueprint, jsonify, request
from ..CyclicXGBC import CyclicXGBClient
from xgboost import XGBClassifier
import requests
import pickle

cyclic_blueprint = Blueprint('cyclic', __name__)
client = CyclicXGBClient()

@cyclic_blueprint.route('/init-encryption', methods=['POST'])
def set_server_public_key():
    data = request.json['data']
    server_public_key = data['server_public_key']
    client_public_key = client.init_encryption(bytes.fromhex(server_public_key))
    return jsonify({'client_public_key': client_public_key.hex()})

@cyclic_blueprint.route('/train', methods=['POST'])
def train():
    encrypted = request.json['encrypted']
    data = client.decrypt_message(bytes.fromhex(encrypted))
    model = data['model']
    num_rounds = data['num_rounds']
    updated_model = client.train(model, num_rounds)
    encrypted = client.encrypt_message({'model': updated_model})
    return jsonify({'encrypted': encrypted.hex()})

@cyclic_blueprint.route('/evaluate', methods=['POST'])
def evaluate():
    encrypted = request.json['encrypted']
    data = client.decrypt_message(bytes.fromhex(encrypted))
    model = data['model']
    tp, fp, fn, tn = client.evaluate(model)
    return jsonify({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
