from flask import Blueprint, jsonify, request
from ..FedXGBC import FedXGBClient, get_data
from ..Models.fedxgb import XGBoostTree
import requests
import numpy as np

fedxgb_blueprint = Blueprint('fedxgb', __name__)
client = FedXGBClient()


@fedxgb_blueprint.route('/get-feature-names', methods=['POST'])
def get_feature_names():
    traindata = get_data()
    feature_names = traindata.columns.tolist()
    feature_names.remove('is_diagnosed')

    serialized = client.serialize_message(feature_names)
    return jsonify({'serialized': serialized.hex()})


@fedxgb_blueprint.route('/init', methods=['POST'])
def init():
    global client

    serialized = request.json.get('serialized')
    data = client.deserialize_message(bytes.fromhex(serialized))
    ordering = data['ordering']

    traindata = get_data()
    if traindata.columns.tolist() != ordering + ['is_diagnosed']:
        traindata = traindata[ordering + ['is_diagnosed']]
    
    client.init_trainer(traindata)
    return jsonify({'success': True})


@fedxgb_blueprint.route('/create-mask', methods=['POST'])
def create_mask():
    print("Creating mask")
    data = request.json['data']
    initializer = data.get('initializer')
    if initializer:
        initializer = np.array(initializer)
    client_list = data.get('client_list')
    first_client = data.get('first_client')
    update = data.get('update')

    if update:
        client.update_mask(np.array(data['delta']))
        print("Updated mask")
        return jsonify({'success': True})


    if first_client is None:
        first_client = client_list.pop(0)

    delta = client.create_mask(initializer)
    
    if client_list:
        next_client = client_list.pop(np.random.randint(0, len(client_list)))
        data = {'data': {
            'initializer': delta.tolist(),
            'client_list': client_list,
            'first_client': first_client,
        }}
    else:
        next_client = first_client
        data = {'data': {'update': True, 'delta': delta.tolist()}}
    
    try:
        response = requests.post(f'{next_client}/fedxgb/create-mask', json=data, verify=False)
        if response.status_code != 200:
            raise Exception("Error in creating masks")
    except Exception as e:
        print(e)
        return jsonify({"error": "Error in creating masks"}), 500

    print("Created mask")
    return jsonify({'success': True})

@fedxgb_blueprint.route('/y', methods=['POST'])
def y():
    ones = int(client.y.sum())
    n_samples = len(client.y)
    data = {'ones': ones, 'n_samples': n_samples}
    serialized = client.serialize_message(data)
    return jsonify({'serialized': serialized.hex()})


@fedxgb_blueprint.route('/feature-importance', methods=['POST'])
def feature_importance():
    serialized = client.serialize_message(client.feature_importance)
    return jsonify({'serialized': serialized.hex()})


@fedxgb_blueprint.route('/quantiles', methods=['POST'])
def quantiles():
    serialized = client.serialize_message(client.quantiles)
    return jsonify({'serialized': serialized.hex()})


@fedxgb_blueprint.route('/binary', methods=['POST'])
def binary():
    serialized = client.serialize_message(client.binary)
    return jsonify({'serialized': serialized.hex()})


@fedxgb_blueprint.route('/histograms', methods=['POST'])
def histograms():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    # data = request.json
    feature_subset = data['feature_subset']
    compute_regions = data['compute_regions']

    histogram = client.compute_histogram(feature_subset, compute_regions)
    histogram = {k: (v + client.mask).tolist() for k, v in histogram.items()}

    serialized = client.serialize_message(histogram)
    return jsonify({'serialized': serialized.hex()})
    # return jsonify(histogram)

@fedxgb_blueprint.route('/set-lr', methods=['POST'])
def set_lr():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    learning_rate = data['learning_rate']
    client.set_learning_rate(learning_rate)
    return jsonify({'success': True})


@fedxgb_blueprint.route('/set-base-y', methods=['POST'])
def set_base_y():
    serialized = request.json['serialized']
    base_y = client.deserialize_message(bytes.fromhex(serialized))
    client.set_base_y(base_y['base_y'])
    return jsonify({'success': True})


@fedxgb_blueprint.route('/set-estimators', methods=['POST'])
def set_estimators():
    data = request.json['data']
    estimators = data['estimators']
    client.set_estimators(estimators)
    return jsonify({'success': True})


@fedxgb_blueprint.route('/set-feature-splits', methods=['POST'])
def set_feature_splits():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    feature_splits = data['feature_splits']
    client.set_feature_splits(feature_splits)
    return jsonify({'success': True})


@fedxgb_blueprint.route('/add-estimator', methods=['POST'])
def add_estimator():
    serialized = request.json['serialized']
    data = client.deserialize_message(bytes.fromhex(serialized))
    estimator = XGBoostTree().from_dict(data['estimator'])
    client.add_estimator(estimator)
    return jsonify({'success': True})

@fedxgb_blueprint.route('/evaluate', methods=['POST'])
def evaluate():
    res = client.evaluate()
    serialized = client.serialize_message(res)
    return jsonify({'serialized': serialized.hex()})