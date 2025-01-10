from .core import TrainClassifier
from .core.utility import Utils
from configparser import ConfigParser
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(42)


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

    return X, y

def finetune_xgb(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series, num_rounds: int):

    if hasattr(model, '_Booster'):

        X = X[model.feature_names_in_]

        dtrain = xgb.DMatrix(X, label=y)
        params = model.get_xgb_params()
        params['seed'] = 42
        model._Booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            xgb_model=model.get_booster(),
            evals=[(dtrain, 'train')],
            verbose_eval=False
        )
    else:
        model.n_estimators = num_rounds
        model.fit(X, y)
    
    return model
    
