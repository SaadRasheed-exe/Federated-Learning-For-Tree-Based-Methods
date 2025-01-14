from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
import pandas as pd
from configparser import ConfigParser
import shutil
from core.Routes import fedxgb_blueprint, agg_blueprint, cyclic_blueprint

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

# Endpoint to check status of the client
@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "active"})

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=7223,
        debug=True,
        ssl_context=('cert.pem', 'key.pem')
    )
