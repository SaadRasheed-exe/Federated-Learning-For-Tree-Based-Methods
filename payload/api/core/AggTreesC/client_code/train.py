from .core import TrainClassifier
from .core.utility import Utils
from configparser import ConfigParser
from datetime import datetime
import logging
import os, shutil
import pandas as pd


def train_model(model, time, feature_names):
    start_time = datetime.now()
    config_file_path = './config.ini'
    config = ConfigParser()
    config.read(config_file_path)

    utils = Utils(config_file_path)
    experiment_path = utils.get_and_make_experiment_path(time)
    print(experiment_path)

    utils.save_config_file(experiment_path)

    classifier = TrainClassifier(experiment_path, config_file_path)
    utils.configure_logging(experiment_path, printing_required=True)

    model = classifier.train_classifier(model, feature_names)

    end_time = datetime.now()

    logging.info(f'It took: {round((end_time-start_time).seconds/60, 3)} minutes')

    if os.path.exists(experiment_path):
        shutil.rmtree('models')

    return model

def get_data():
    config_file_path = './config.ini'
    config = ConfigParser()
    config.read(config_file_path)

    utils = Utils(config_file_path)
    experiment_path = utils.get_and_make_experiment_path(datetime.now())
    print(experiment_path)

    utils.save_config_file(experiment_path)

    classifier = TrainClassifier(experiment_path, config_file_path)
    utils.configure_logging(experiment_path, printing_required=True)

    x_train, x_test, y_train, y_test = classifier.process.get_data(is_survival=False)

    X = pd.concat([x_train, x_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)

    if os.path.exists(experiment_path):
        shutil.rmtree('models')

    return X, y