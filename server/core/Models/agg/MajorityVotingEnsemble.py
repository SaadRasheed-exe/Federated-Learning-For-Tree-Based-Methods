from collections import Counter
from typing import Literal, Dict, Any
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import ipdb


class MajorityVotingEnsemble:
    def __init__(self, models: Dict, model_weightage: Dict = None):
        self.models = models
        if model_weightage is None:
            self.model_weightage = {k: 1 for k in models.keys()}
        else:
            self.model_weightage = model_weightage
        
        self._feature_names_in_ = MajorityVotingEnsemble._get_model_feature_names(list(self.models.values())[0])
    
    def __repr__(self):
        res = f'MajorityVotingEnsemble({self.models})'
        return res
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._feature_names_in_ = X.columns.tolist()
        for _, model in self.models.items():
            model.fit(X, y)
        
    def predict(self, X: pd.DataFrame, individual: bool = False):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_names_in_)
        elif isinstance(X, pd.DataFrame):
            X = X[self._feature_names_in_]
        else:
            raise ValueError('Invalid input type')

        predictions = []
        for _, model in self.models.items():
            # cols = MajorityVotingEnsemble._get_model_feature_names(model)
            # X = X[cols]
            preds = model.predict(X)
            predictions.append(preds)
                
        if individual:
            return np.array(predictions)

        majority_votes = []
        for i in range(len(X)):
            votes = Counter()
            for j, (_, weight) in enumerate(self.model_weightage.items()):
                votes[predictions[j][i]] += weight
            majority_votes.append(votes.most_common(1)[0][0])
        
        return np.array(majority_votes)
    
    def score(self, X: pd.DataFrame, y: pd.Series, metric: Literal['f1', 'acc'] = 'f1'):
        predictions = self.predict(X)
        
        if metric == 'f1':
            return f1_score(y, predictions)
        elif metric == 'acc':
            return accuracy_score(y, predictions)
        else:
            raise ValueError('Invalid metric')
    
    def serialize(self):
        return {
            'models': self.models,
            'model_weightage': self.model_weightage
        }
    
    @property
    def feature_names_in_(self):
        if self._feature_names_in_ is None:
            self._feature_names_in_ = MajorityVotingEnsemble._get_model_feature_names(list(self.models.values())[0])

        return self._feature_names_in_
    
    @feature_names_in_.setter
    def feature_names_in_(self, value):
        self._feature_names_in_ = value

    @staticmethod
    def _get_model_feature_names(model: Any):
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_
        elif isinstance(model, (XGBClassifier, XGBRegressor)):
            return model.get_booster().feature_names
        elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
            return model.booster_.feature_name()
        else:
            raise ValueError(f'{type(model)} models are not supported')
    
    def shap_values(self, X: pd.DataFrame):
        total_shap = None
        weightages = []

        for _id, model in self.models.items():

            weightages.append(self.model_weightage[_id])
            
            if isinstance(model, (XGBClassifier, XGBRegressor)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
                explainer = shap.TreeExplainer(model.booster_)
            elif isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                raise ValueError(f'{type(model)} models are not supported')
            
            if total_shap is not None:
                if total_shap.shape != explainer.shap_values(X).shape:
                    raise ValueError('Shape mismatch in shap values')
                shap_values = explainer.shap_values(X)
                total_shap += np.array(shap_values) * self.model_weightage[_id]
            else:
                shap_values = explainer.shap_values(X)
                total_shap = np.array(shap_values) * self.model_weightage[_id]
        
        # weighted average of shap values
        total_shap /= sum(weightages)
        return total_shap[:, :, 1]
        