import numpy as np
import pandas as pd
from .FedXGBCTrainer import FedXGBCTrainer
from ..Utility import BaseClient

class FedXGBClient(BaseClient):

    """
    A class representing a client in the federated XGBoost framework.

    Attributes:
        X (np.ndarray): The feature matrix for the client.
        y (np.ndarray): The target vector for the client.
        n_features (int): The number of features in the feature matrix.
        samples (int): The number of samples in the feature matrix.
        estimators (list): A list of estimators (trees) for the client.
        base_y (float): The base prediction value.
        learning_rate (float): The learning rate for the model.
        n_quantiles (int): The number of quantiles for the features.
        quantiles (dict): The quantiles for each feature.
        feature_importance (dict): The importance of each feature.
        binary (dict): A dictionary containing binary features.
        histogram (Histogram): The histogram object for the client.
        y_preds (np.ndarray): The predictions for the target vector.

    Methods:
        __init__(X, y, get_importance=True): Initializes the client with the given data and parameters.
        compute_histogram(feature_splits, compute_regions): Computes the histogram for the client.
        get_quantiles(): Returns the quantiles of the features.
        get_feature_importance(): Calculates the feature importance of the model.
        get_binary_features(): Returns a list of binary features.
        sigmoid(x): Computes the sigmoid function.
        grad(preds, labels): Computes the first order gradient of the log loss.
        hess(preds, labels): Computes the second order gradient of the log loss.
        predict(X): Predicts the target values for the given feature matrix.
    """ 
    def __init__(self):
        super().__init__()
        self.trainer = None

    def init_trainer(self, traindata: pd.DataFrame, get_importance: bool = False):
        self.trainer = FedXGBCTrainer(traindata, get_importance)

    def compute_histogram(self, features_subset, compute_regions):
        return self.trainer.compute_histogram(features_subset, compute_regions)

    def create_mask(self, initializer):
        self.mask = 5 * np.random.randn(1)
        delta = initializer - self.mask
        return delta
    
    def update_mask(self, delta):
        self.mask += delta
    
    def set_learning_rate(self, learning_rate):
        self.trainer.learning_rate = learning_rate
    
    def set_base_y(self, base_y):
        self.trainer.base_y = base_y

    def set_estimators(self, estimators):
        self.trainer.estimators = estimators

    def set_feature_splits(self, feature_splits):
        self.trainer.init_histogram(feature_splits)
    
    def add_estimator(self, estimator):
        self.trainer.estimators.append(estimator)
    
    def evaluate(self):
        return self.trainer.evaluate()

    @property
    def feature_importance(self):
        return self.trainer.feature_importance

    @property
    def y(self):
        return self.trainer.y
    
    @property
    def quantiles(self):
        return self.trainer.quantiles
    
    @property
    def binary(self):
        return self.trainer.binary