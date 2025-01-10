from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
import requests
import json
from charset_normalizer import detect
import pandas as pd
from configparser import ConfigParser
import shutil
from core.Routes import fedxgb_blueprint, agg_blueprint, cyclic_blueprint

SERVER_URL = "http://172.16.105.129"
SERVER_PORT = 7223

server_url = f'{SERVER_URL}:{SERVER_PORT}'

def send_info():
    with open('ip.txt', 'rb') as f:
        raw_data = f.read()
    result = detect(raw_data)
    encoding = result['encoding']
    with open('ip.txt', 'r', encoding=encoding) as f:
        ip = f.readlines()[8]
        ip = ip.split(" ")[-1]
    with open('data/info.json', 'r') as f:
        state = json.load(f)['state']
    requests.post(f"{SERVER_URL}:{SERVER_PORT}/gather-info", json={"ip": ip, "state": state})
    print('Info sent to server.')

# send_info()

app = Flask(__name__)
app.register_blueprint(fedxgb_blueprint, url_prefix='/fedxgb')
app.register_blueprint(agg_blueprint, url_prefix='/agg')
app.register_blueprint(cyclic_blueprint, url_prefix='/cyclic')

# Endpoint to receive a file from the server and save it in specified subdirectory
@app.route('/upload', methods=['POST'])
def upload_file():
    subdirectory = request.form.get('subdirectory', '')
    format_ = request.form.get('format', 'file')

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if subdirectory:
        os.makedirs(os.path.join(os.getcwd(), subdirectory), exist_ok=True)

    file.save(os.path.join(os.getcwd(), subdirectory, file.filename))

    if format_ == 'zip':
        shutil.unpack_archive(os.path.join(os.getcwd(), subdirectory, file.filename), os.path.join(os.getcwd(), subdirectory, file.filename.split('.')[0]))
        os.remove(os.path.join(os.getcwd(), subdirectory, file.filename))

    return jsonify({"message": "File uploaded successfully"})

# Endpoint to run a command on the client
@app.route('/run-command', methods=['POST'])
def run_command():
    data = request.get_json()
    if not data or 'command_args' not in data:
        return jsonify({"error": "Missing command arguments."}), 400

    command_args = data['command_args']

    try:
        result = subprocess.run(command_args, capture_output=True, text=True)
        return jsonify({"output": result.stdout, "errors": result.stderr, "return_code": result.returncode})
    except subprocess.CalledProcessError as e:
        # logging.error(f"Command execution failed: {e}")
        return jsonify({"error": str(e), "return_code": 500}), 500
    except Exception as e:
        # logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e), "return_code": 500}), 500

# Endpoint to send a requested file back to the server
@app.route('/download', methods=['GET'])
def download_file():
    subdirectory = request.args.get('subdirectory', '')
    filename = request.args.get('filename', '')

    if not filename:
        return jsonify({"error": "Missing filename"}), 400

    file_path = os.path.join(os.getcwd(), subdirectory, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File {filename} does not exist in {subdirectory}"}), 404

    return send_from_directory(os.path.join(os.getcwd(), subdirectory), filename)

# Endpoint to send data stats to the server
@app.route('/send-stats', methods=['GET'])
def send_stats():

    if not os.path.exists('config.ini'):
        return jsonify({})

    config = ConfigParser()
    config.read('config.ini')
    disease = config.get('BASE_SETTINGS', 'modelling_which_disease')
    print('Reading data stats for disease:', disease)

    data_path = os.path.join('data', disease)
    data_stats = {}
    for file in os.listdir(data_path):
        if not file.endswith('.csv'):
            continue
        class_name = 'diagnosed' if 'diagnosed' in file else 'normal'
        data_stats[class_name] = {}
        data = pd.read_csv(os.path.join(data_path, file))
        data_stats[class_name]['columns'] = data.columns.tolist()
        data_stats[class_name]['dtypes'] = data.dtypes.apply(str).tolist()
        data_stats[class_name]['n_samples'] = data.shape[0]
        data_stats[class_name]['n_features'] = data.shape[1]
    
    return jsonify(data_stats)


# Endpoint to start training
@app.route('/train', methods=['POST'])
def train():

    if not os.path.exists('config.ini'):
        return jsonify({"error": "Config file not found."}), 500

    command_args = ['pyenv/bin/python', 'train.py']
    try:
        result = subprocess.run(command_args, capture_output=True, text=True)
        
        pre_models_path = os.path.join(os.getcwd(), 'pretrained_models')
        if not os.path.exists(pre_models_path):
            os.makedirs(pre_models_path)
        
        # upload the trained models to the server and specify the sub directory
        for file in os.listdir(pre_models_path):
            with open(os.path.join(pre_models_path, file), 'rb') as f:
                response = requests.post(f"{server_url}/upload", files={'file': f}, data={'subdirectory': 'pretrained_models'})
                print(response.json())

        return jsonify({"output": result.stdout, "errors": result.stderr, "return_code": result.returncode})
    except Exception as e:
        return jsonify({"errors": str(e)}), 500

# Endpoint to check status of the client
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "active"})
    
# Endpoint to reset the client process by deleting the models and pretrained_models directories
@app.route('/reset', methods=['POST'])
def reset():
    try:
        if os.path.exists('models'):
            os.system('rm -r models')
        if os.path.exists('pretrained_models'):
            os.system('rm -r pretrained_models')
        for file in [file for file in os.listdir('.') if file.endswith('.csv')]:
            os.remove(file)
        return jsonify({"message": "Client reset successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7223, debug=True)
