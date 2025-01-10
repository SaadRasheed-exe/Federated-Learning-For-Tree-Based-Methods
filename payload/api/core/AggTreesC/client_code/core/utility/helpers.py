from sklearn.tree import BaseDecisionTree
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def get_model_feature_names(model):
    if isinstance(model, BaseDecisionTree):
        return model.feature_names_in_
    elif isinstance(model, (XGBClassifier, XGBRegressor)):
        return model.get_booster().feature_names
    elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
        return model.booster_.feature_name()
    else:
        raise ValueError('Model not supported')