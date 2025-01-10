from collections import Counter
from typing import List, Literal, Dict
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


class MajorityVotingEnsemble:
    def __init__(self, models: Dict, model_weightage: Dict = None):
        self.models = models
        if model_weightage is None:
            self.model_weightage = {k: 1 for k in models.keys()}
        else:
            self.model_weightage = model_weightage
    
    def __repr__(self):
        res = f'MajorityVotingEnsemble({self.models})'
        return res

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for _, model in self.models.items():
            model.fit(X, y)
        
    def predict(self, X: pd.DataFrame, individual: bool = False):

        predictions = [model.predict(X[MajorityVotingEnsemble._get_model_feature_names(model)]) for _, model in self.models.items()]

        if individual:
            return predictions

        majority_votes = []
        for i in range(len(X)):
            votes = Counter()
            for j, (_, weight) in enumerate(self.model_weightage.items()):
                votes[predictions[j][i]] += weight
            majority_votes.append(votes.most_common(1)[0][0])
        
        return majority_votes
    
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

    @staticmethod
    def _get_model_feature_names(model):
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_
        elif isinstance(model, (XGBClassifier, XGBRegressor)):
            return model.get_booster().feature_names
        elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
            return model.booster_.feature_name()
        else:
            raise ValueError(f'{type(model)} models are not supported')
