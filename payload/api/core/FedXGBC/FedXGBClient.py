import numpy as np
import pandas as pd
from ..Utility import BaseClient
from xgboost import XGBClassifier
from ..Models.fedxgb import Histogram, FedXGBoostEnsemble

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
    def init_trainer(self, traindata: pd.DataFrame, get_importance: bool = False):
        X = traindata.drop('is_diagnosed', axis=1)
        y = traindata['is_diagnosed']

        self.feature_names = X.columns.tolist()
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.Series) else y
        self.n_features = X.shape[1]
        self.samples = X.shape[0]

        self.n_quantiles = int(np.ceil(self.X.shape[0] / 10) + 2)

        if get_importance:
            self.feature_importance = self._get_feature_importance()
        self.binary = self._get_binary_features()
        self.quantiles = self._get_quantiles()
        self.estimators = []
        self.learning_rate = None
        self.base_y = None
        self.y_preds = None
        self.histogram = None

    def _get_feature_importance(self):
        '''
        Calculates the feature importance of the model.
        '''
        # Train an XGBoost model
        model = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.3)
        model.fit(self.X, self.y)

        # Get the feature importance from the model
        feature_importance = model.feature_importances_
        return {i: float(feature_importance[i]*self.samples) for i in range(len(feature_importance))}

    def _get_binary_features(self):
        '''
        Returns a list of binary features.
        '''
        binary = {} # dictionary containing features with n_unique <= 2 {feature_index: n_unique}
        # get a list of binary features from self.client_X, it is a numpy array
        for i in range(self.X.shape[1]):
            unique_vals = np.unique(self.X[:, i])
            if len(unique_vals) <= 2:
                binary[i] = unique_vals.tolist()
        return binary

    def _get_quantiles(self):
        '''
        Returns the quantiles of the features.
        '''
        quantiles = {}
        for i in range(self.X.shape[1]):
            if i not in self.binary:
                quantiles[i] = np.quantile(self.X[:, i], q=np.linspace(0, 1, self.n_quantiles)).tolist()
            else:
                quantiles[i] = list(self.binary[i])
        return quantiles

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _grad(preds, labels):
        preds = FedXGBClient._sigmoid(preds)
        return(preds - labels)
    
    @staticmethod
    def _hess(preds):
        preds = FedXGBClient._sigmoid(preds)
        return(preds * (1 - preds))

    def create_mask(self, initializer):
        self.mask = 5 * np.random.randn(1)
        delta = initializer - self.mask
        return delta
    
    def update_mask(self, delta):
        self.mask += delta

    def set_feature_splits(self, feature_splits):
        self.histogram = Histogram(feature_splits=feature_splits)
        self.histogram.fit(self.X)
    
    def compute_histogram(self, features_subset, compute_regions):
        if self.estimators:
            self.y_preds += self.learning_rate * self.estimators[-1].predict(self.X)
        elif self.base_y is not None:
            self.y_preds = np.full((self.X.shape[0], 1), self.base_y).flatten().astype('float64')
        else:
            raise ValueError("No initial predictions available.")
       
        Grads = self._grad(self.y_preds, self.y)
        Hess = self._hess(self.y_preds)
        return self.histogram.compute_histogram(Grads, Hess, features_subset, compute_regions)
    
    def evaluate(self):
        self.final_model = FedXGBoostEnsemble(
            estimators=self.estimators,
            base_y=self.base_y,
            learning_rate=self.learning_rate,
            feature_names=self.feature_names,
        )

        y_preds = self.final_model.predict(self.X)
        tp = np.sum((y_preds == 1) & (self.y == 1))
        tn = np.sum((y_preds == 0) & (self.y == 0))
        fp = np.sum((y_preds == 1) & (self.y == 0))
        fn = np.sum((y_preds == 0) & (self.y == 1))

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }