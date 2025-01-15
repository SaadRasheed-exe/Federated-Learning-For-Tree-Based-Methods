from .XGBoostTree import XGBoostTree
from typing import List
import numpy as np
import pandas as pd

class FedXGBoostEnsemble:
    """
    A FedXGBoostEnsemble class that implements an ensemble model using a collection of XGBoost trees.
    The ensemble aggregates the predictions of each tree with weighted sums and applies a sigmoid transformation.

    Attributes:
    - estimators (list): A list of XGBoostTree objects used as estimators in the ensemble.
    - base_y (float): The base prediction value for the ensemble.
    - learning_rate (float): The learning rate used to weight the contribution of each tree's prediction.
    - feature_names_in_ (list, optional): The feature names used in the model.
    - feature_importance (list, optional): The feature importance values for each feature.
    
    Methods:
    - __call__: Allows calling the instance as a function, which forwards to `predict_proba` or `predict` based on the `proba` argument.
    - sigmoid: Applies the sigmoid activation function.
    - predict_proba: Predicts the probabilities for the given input data.
    - predict: Predicts the class labels based on the probabilities.
    """

    def __init__(
            self,
            estimators: List[XGBoostTree],
            base_y: float,
            learning_rate: float = 0.1,
            feature_names: List[str] = None,
            feature_importance: List[float] = None
        ):
        """
        Initializes the FedXGBoostEnsemble with a list of estimators and other ensemble parameters.
        
        Args:
        - estimators (list): A list of XGBoostTree estimators to be used in the ensemble.
        - base_y (float): The base prediction value for the ensemble.
        - learning_rate (float, optional): The learning rate that adjusts the contribution of each tree.
          Default is 0.1.
        - feature_names (list, optional): List of feature names used by the model.
        - feature_importance (list, optional): List of feature importance values used to adjust predictions.
        """
        self.estimators = estimators
        self.base_y = base_y
        self.learning_rate = learning_rate
        self.feature_names_in_ = feature_names
        self.feature_importance = feature_importance
    
    @staticmethod
    def sigmoid(x):
        """
        Applies the sigmoid activation function to the input value or array.
        
        Args:
        - x (numpy.ndarray or float): The input value(s).
        
        Returns:
        - numpy.ndarray or float: The output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def __call__(self, X: np.ndarray, proba: bool = False):
        """
        Calls the instance to predict either probabilities or class labels based on the `proba` flag.
        
        Args:
        - X (numpy.ndarray or pd.DataFrame): The input feature data.
        - proba (bool, optional): If True, returns predicted probabilities; otherwise, returns class labels.
        
        Returns:
        - numpy.ndarray: The predicted class labels or probabilities.
        """
        return self.predict_proba(X) if proba else self.predict(X)

    def predict_proba(self, X: np.ndarray):
        """
        Predicts the class probabilities for the given input data using the ensemble of estimators.
        
        Args:
        - X (numpy.ndarray or pd.DataFrame): The input feature data.
        
        Returns:
        - numpy.ndarray: The predicted probabilities for each input.
        """
        # Convert DataFrame to numpy if necessary
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        
        # Initialize predictions with the base prediction value for all samples
        pred = np.full(
            (X.shape[0], 1), 
            self.base_y
        ).flatten().astype('float64')

        # Sum the predictions of each estimator, weighted by the learning rate
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        # Apply the sigmoid function to the final predictions
        return self.sigmoid(
            np.full(
                (X.shape[0], 1),
                1
            ).flatten().astype('float64') + pred
        )
    
    def predict(self, X):
        """
        Predicts the class labels for the given input data by applying a threshold on predicted probabilities.
        
        Args:
        - X (numpy.ndarray or pd.DataFrame): The input feature data.
        
        Returns:
        - numpy.ndarray: The predicted class labels (0 or 1).
        """
        # Get predicted probabilities
        predicted_probas = self.predict_proba(X)
        
        # Predict the class labels based on the mean probability threshold
        preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        
        return preds
