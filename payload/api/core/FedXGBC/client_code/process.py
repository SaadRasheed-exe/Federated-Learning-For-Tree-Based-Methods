from .core.utility import Utils, ProcessDataset
import pandas as pd
import numpy as np
from configparser import ConfigParser

np.random.seed(42)

def get_data():
    config_file_path = './config.ini'
    utils = Utils(config_file_path)
    experiment_path = utils.get_and_make_experiment_path()
    process = ProcessDataset(experiment_path, config_file_path)

    config = ConfigParser()
    config.read(config_file_path)
    val_required = eval(config.get('BASE_SETTINGS', 'validation_set_required'))

    if val_required:
        x_train, x_test, x_val, y_train, y_test,  y_val = process.get_data(is_survival=False)

        traindata = pd.concat([x_train, y_train], axis=1)
        testdata = pd.concat([x_test, y_test], axis=1)
        valdata = pd.concat([x_val, y_val], axis=1)

        traindata = pd.concat([traindata, testdata, valdata], axis=0, ignore_index=True)

        # traindata.to_csv(f"./traindata.csv", index=False)

    else:
        x_train, x_test, y_train, y_test = process.get_data(is_survival=False)
        
        traindata = pd.concat([x_train, y_train], axis=1)
        testdata = pd.concat([x_test, y_test], axis=1)

        traindata = pd.concat([traindata, testdata], axis=0, ignore_index=True)

        # traindata.to_csv(f"./traindata.csv", index=False)
    
    return traindata