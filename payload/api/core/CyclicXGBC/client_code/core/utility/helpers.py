from sklearn.tree import BaseDecisionTree
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import xgboost as xgb


def get_model_feature_names(model):
    if isinstance(model, BaseDecisionTree):
        return model.feature_names_in_
    elif isinstance(model, (XGBClassifier, XGBRegressor)):
        return model.get_booster().feature_names
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        return model.booster_.feature_name()
    else:
        raise ValueError('Model not supported')

def finetune_xgb(model, dtrain, dtest, num_rounds):
    params = model.get_xgb_params()
    params['seed'] = 42
    model._Booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        xgb_model=model.get_booster(),
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval=False
    )
    return model