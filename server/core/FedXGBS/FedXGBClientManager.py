from typing import Any, Dict, List
import numpy as np
from ..Utility import BaseClientManager

class FedXGBClientManager(BaseClientManager):
    def fetch_feature_names(self):
        """
        Fetch feature names from all active clients.
        Returns:
            feature_names (list): A list of feature names from the clients.
        """
        for client_id in self.active_clients:
            response = self._communicate(client_id, 'fedxgb/get-feature-names', serialize=False)
            if response:
                return response
            else:
                self._handle_client_failure(client_id)

    def init_clients(self, feature_names: list):
        """
        Initialize communication with all active clients, exchanging public keys and preparing the system.
        """
        data = {'ordering': feature_names}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/init', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
    
    def init_masks(self):
        """
        Initialize the masks for all active clients.
        """
        initializer = [0]
        for client_id in self.active_clients:
            try:
                client_list = [('https://' + url + f':{self.CLIENT_PORT}') for c, url in self.clients.items() if c in self.active_clients]
                data = {
                    'initializer': initializer,
                    'client_list': client_list,
                }
                response = self._communicate(client_id, 'fedxgb/create-mask', data, serialize=False)
                if response:
                    break
                else:
                    self._handle_client_failure(client_id)
            except Exception as e:
                print(f"Error initializing mask for client {client_id}: {str(e)}")
                self._handle_client_failure(client_id)

    def get_binary(self, num_features: int):
        """
        Retrieve the binary data from all active clients.
        Returns:
            binary_data (dict): A dictionary of binary data from the clients.
        """
        binary_data = {i: set() for i in range(num_features)}
        result = self._execute_in_threads(1, self._communicate, 'fedxgb/binary')
        for client_id, response in result.items():
            if response:
                for feature in response:
                    binary_data[feature] = binary_data[feature].union(set(response[feature]))
            else:
                self._handle_client_failure(client_id)
        return {int(k): list(v) for k, v in binary_data.items() if len(list(v)) == 2}

    def set_feature_splits(self, feature_splits: Dict[int, List[int]]):
        """
        Set the feature splits for all active clients.
        Args:
            feature_splits (dict): The feature splits to set.
        """
        data = {'feature_splits': feature_splits}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-feature-splits', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def get_y(self):
        """
        Retrieve the binary data from all active clients.
        Returns:
            binary_data (dict): A dictionary of binary data from the clients.
        """
        ones = 0
        n_samples = 0
        result = self._execute_in_threads(1, self._communicate, 'fedxgb/y')
        for client_id, response in result.items():
            if response:
                ones += response['ones']
                n_samples += response['n_samples']
            else:
                self._handle_client_failure(client_id)
        
        return ones, n_samples
    
    def get_quantiles(self):
        """
        Retrieve the quantiles from all active clients.
        Returns:
            quantiles (dict): A dictionary of quantiles from the clients.
        """
        quantiles = {}
        result = self._execute_in_threads(1, self._communicate, 'fedxgb/quantiles')
        for client_id, response in result.items():
            if response:
                for feature in response:
                    if feature not in quantiles:
                        quantiles[feature] = []
                    quantiles[feature] += response[feature]
            else:
                self._handle_client_failure(client_id)

        return {int(k): v for k, v in quantiles.items()}

    def get_feature_importance(self):
        """
        Retrieve the feature importance from all active clients.
        Returns:
            feature_importance (dict): A dictionary of feature importance values from the clients.
        """
        feature_importance = {}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/feature-importance')
        for client_id, response in result.items():
            if response:
                for feature in response:
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    feature_importance[feature] += response[feature]
            else:
                self._handle_client_failure(client_id)
        return {int(k): v for k, v in feature_importance.items()}
    
    def set_base_y(self, base_y: int):
        """
        Set the base y value for all active clients.
        Args:
            base_y (int): The base y value to set.
        """
        data = {'base_y': base_y}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-base-y', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
    
    def set_learning_rate(self, learning_rate: float):
        """
        Set the learning rate for all active clients.
        Args:
            learning_rate (float): The learning rate to set.
        """
        data = {'learning_rate': learning_rate}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-lr', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
    
    def set_estimators(self, estimators: list):
        """
        Set the number of estimators for all active clients.
        Args:
            estimators (int): The number of estimators to set.
        """
        data = {'estimators': estimators}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-estimators', data, serialize=False)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
    
    def add_estimator(self, estimator: Any):
        """
        Add an estimator to all active clients.
        Args:
            estimator (int): The estimator to add.
        """
        data = {'estimator': estimator}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/add-estimator', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
    
    def get_histograms(self, feature_subset: List, compute_regions: bool):
        """
        Retrieve the histogram from all active clients.
        Args:
            feature_splits (dict): The feature splits to use for histogram computation.
            compute_regions (bool): Whether to compute regions.
        Returns:
            histogram (dict): A dictionary of histograms from the clients.
        """
        histogram: Dict[str, Dict[str, float]] = {}
        data = {'feature_subset': feature_subset, 'compute_regions': compute_regions}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/histograms', data)
        for client_id, response in result.items():
            if response:
                for key in response:
                    if key not in histogram:
                        histogram[key] = {}
                    histogram[key][client_id] = response[key]
            else:
                self._handle_client_failure(client_id)

        histogram = {k: np.sum(list(v.values()), axis=0) for k, v in histogram.items()}
        return histogram
    
    def evaluate(self):
        """
        Evaluate the model on all active clients.
        Returns:
            evaluation (dict): A dictionary of evaluation results from the clients.
        """
        eval_results = {}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/evaluate')
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)
            else:
                for key, val in response.items():
                    if key not in eval_results:
                        eval_results[key] = 0
                    eval_results[key] += val
        return eval_results