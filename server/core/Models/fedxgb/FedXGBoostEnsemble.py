from .XGBoostTree import XGBoostTree
from typing import List
import numpy as np
import pandas as pd

class FedXGBoostEnsemble:
    def __init__(
            self,
            estimators: List[XGBoostTree],
            base_y: float,
            learning_rate: float = 0.1,
            feature_names: List[str] = None,
            feature_importance: List[float] = None
        ):
        self.estimators = estimators
        self.base_y = base_y
        self.learning_rate = learning_rate
        self.feature_names_in_ = feature_names
        self.feature_importance = feature_importance
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def __call__(self, X: np.ndarray, proba: bool = False):
        return self.predict_proba(X) if proba else self.predict(X)

    def predict_proba(self, X: np.ndarray):
        # pred is a 1D array of all values equal to self.base_y of shape (X.shape[0],): [y, y, y, y ...]
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        pred = np.full(
            (X.shape[0], 1), 
            self.base_y
        ).flatten().astype('float64')

        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return(self.sigmoid(
            np.full(
                (X.shape[0], 1),
                1
            ).flatten().astype('float64') + pred
        ))
    
    def predict(self, X):
        predicted_probas = self.predict_proba(X)
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return(preds)