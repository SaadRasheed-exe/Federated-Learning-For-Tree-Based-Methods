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


class MajorityVotingEnsemble:
    """
    A Majority Voting Ensemble class that combines predictions from multiple models 
    using a weighted majority voting approach.

    Attributes:
    - models (dict): A dictionary of models to be used in the ensemble.
    - model_weightage (dict): A dictionary of weightages associated with each model.
    - feature_names_in_ (list): List of feature names in the input data.

    Methods:
    - __repr__: Returns a string representation of the ensemble.
    - fit: Trains the models on the provided dataset.
    - predict: Makes predictions using the trained models, optionally returning individual model predictions.
    - score: Calculates the performance of the ensemble based on a chosen metric ('f1' or 'acc').
    - serialize: Serializes the model and its associated data for storage.
    - shap_values: Computes the SHAP (SHapley Additive exPlanations) values for the features used by the ensemble models.
    """
    
    def __init__(self, models: Dict, model_weightage: Dict = None):
        """
        Initializes the MajorityVotingEnsemble with models and optional model weightages.
        
        Args:
        - models (dict): A dictionary of models to be used in the ensemble.
        - model_weightage (dict, optional): A dictionary of weightages for the models.
          If None, all models are assigned equal weightage.
        """
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
        """
        Fits all models in the ensemble to the provided training data.
        
        Args:
        - X (pd.DataFrame): The input feature data.
        - y (pd.Series): The target labels.
        """
        self._feature_names_in_ = X.columns.tolist()
        for _, model in self.models.items():
            model.fit(X, y)
        
    def predict(self, X: pd.DataFrame, individual: bool = False):
        """
        Predicts the target labels for the given input data using the ensemble.
        
        Args:
        - X (pd.DataFrame or np.ndarray): The input feature data.
        - individual (bool, optional): Whether to return individual model predictions.
        
        Returns:
        - np.ndarray: Predicted labels (majority voting results).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self._feature_names_in_)
        elif isinstance(X, pd.DataFrame):
            X = X[self._feature_names_in_]
        else:
            raise ValueError('Invalid input type')

        predictions = []
        for _, model in self.models.items():
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
        """
        Evaluates the performance of the ensemble based on a specified metric ('f1' or 'acc').

        Args:
        - X (pd.DataFrame): The input feature data.
        - y (pd.Series): The true target labels.
        - metric (str, optional): The metric to evaluate the model on ('f1' or 'acc').
        
        Returns:
        - float: The evaluation score based on the specified metric.
        """
        predictions = self.predict(X)
        
        if metric == 'f1':
            return f1_score(y, predictions)
        elif metric == 'acc':
            return accuracy_score(y, predictions)
        else:
            raise ValueError('Invalid metric')
    
    def serialize(self):
        """
        Serializes the models and their weightages for storage.
        
        Returns:
        - dict: A dictionary containing the models and their weightages.
        """
        return {
            'models': self.models,
            'model_weightage': self.model_weightage
        }
    
    @property
    def feature_names_in_(self):
        """
        Returns the feature names in the input data.
        """
        if self._feature_names_in_ is None:
            self._feature_names_in_ = MajorityVotingEnsemble._get_model_feature_names(list(self.models.values())[0])

        return self._feature_names_in_
    
    @feature_names_in_.setter
    def feature_names_in_(self, value):
        """
        Sets the feature names in the input data.
        """
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
        """
        Computes the SHAP (SHapley Additive exPlanations) values for the ensemble models.
        
        Args:
        - X (pd.DataFrame): The input feature data.
        
        Returns:
        - np.ndarray: The SHAP values for the features, weighted by model importance.
        
        Raises:
        - ValueError: If a model type is not supported for SHAP calculations.
        """
        total_shap = None
        weightages = []

        for _id, model in self.models.items():

            weightages.append(self.model_weightage[_id])
            
            if isinstance(model, (XGBClassifier, DecisionTreeClassifier, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, (LGBMClassifier)):
                explainer = shap.TreeExplainer(model.booster_)
            else:
                raise ValueError(f'{type(model)} models are not supported')
            
            shap_values = explainer.shap_values(X)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            if total_shap is not None:
                if total_shap.shape != shap_values.shape:
                    raise ValueError('Shape mismatch in shap values')
                total_shap += np.array(shap_values) * self.model_weightage[_id]
            else:
                total_shap = np.array(shap_values) * self.model_weightage[_id]
        
        # weighted average of shap values
        total_shap /= sum(weightages)
        return total_shap
