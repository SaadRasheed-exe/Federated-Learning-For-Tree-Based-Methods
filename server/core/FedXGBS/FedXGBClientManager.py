from typing import Any, Dict, List
import numpy as np
from ..Utility import BaseClientManager

class FedXGBClientManager(BaseClientManager):
    """
    Manages communication and operations with clients in a federated XGBoost setup.
    This class provides methods to fetch feature names, initialize clients, train models,
    and aggregate results from clients in a federated learning framework.
    """

    def fetch_feature_names(self):
        """
        Fetches the feature names from all active clients in the federated system.

        Returns:
            List: A list of feature names if successfully retrieved from any client.

        Raises:
            Exception: If communication with a client fails.
        """
        for client_id in self.active_clients:
            response = self._communicate(client_id, 'fedxgb/get-feature-names', serialize=False)
            if response:
                return response
            else:
                self._handle_client_failure(client_id)

    def init_clients(self, feature_names: list):
        """
        Initializes clients with the provided feature names.

        Args:
            feature_names (list): A list of feature names to initialize the clients with.

        Raises:
            Exception: If communication with any client fails during initialization.
        """
        data = {'ordering': feature_names}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/init', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def init_masks(self):
        """
        Initializes masks for all active clients.

        Raises:
            Exception: If an error occurs during the initialization of masks or communication with a client.
        """
        initializer = [0]
        for client_id in self.active_clients:
            try:
                # Construct the client list URL for communication.
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
        Fetches binary data from the clients based on the specified number of features.

        Args:
            num_features (int): The number of features for which binary data is requested.

        Returns:
            dict: A dictionary where keys are feature indices and values are lists of binary values.

        Raises:
            Exception: If communication with any client fails.
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
        Sets the feature splits for the federated system.

        Args:
            feature_splits (Dict[int, List[int]]): A dictionary mapping feature indices to the corresponding splits.
        
        Raises:
            Exception: If communication with any client fails during the process.
        """
        data = {'feature_splits': feature_splits}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-feature-splits', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def get_y(self):
        """
        Fetches the target variable information (number of ones and total number of samples) from the clients.

        Returns:
            tuple: A tuple containing the count of ones and the total number of samples.

        Raises:
            Exception: If communication with any client fails.
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
        Fetches quantile data for the features from the clients.

        Returns:
            dict: A dictionary containing quantiles for each feature from the clients.

        Raises:
            Exception: If communication with any client fails.
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
        Fetches the feature importance values from the clients.

        Returns:
            dict: A dictionary containing feature importance values aggregated from all clients.

        Raises:
            Exception: If communication with any client fails.
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
        Sets the base value for the target variable on the clients.

        Args:
            base_y (int): The base value for the target variable to be set on clients.
        
        Raises:
            Exception: If communication with any client fails.
        """
        data = {'base_y': base_y}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-base-y', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def set_learning_rate(self, learning_rate: float):
        """
        Sets the learning rate for the federated XGBoost model.

        Args:
            learning_rate (float): The learning rate to be set for the clients.

        Raises:
            Exception: If communication with any client fails.
        """
        data = {'learning_rate': learning_rate}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-lr', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def set_estimators(self, estimators: list):
        """
        Sets the list of estimators (models) for the federated XGBoost system.

        Args:
            estimators (list): A list of estimators to be set on the clients.

        Raises:
            Exception: If communication with any client fails.
        """
        data = {'estimators': estimators}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/set-estimators', data, serialize=False)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def add_estimator(self, estimator: Any):
        """
        Adds an estimator (model) to the federated XGBoost system.

        Args:
            estimator (Any): The estimator to be added to the system.

        Raises:
            Exception: If communication with any client fails.
        """
        data = {'estimator': estimator}
        result = self._execute_in_threads(4, self._communicate, 'fedxgb/add-estimator', data)
        for client_id, response in result.items():
            if not response:
                self._handle_client_failure(client_id)

    def get_histograms(self, feature_subset: List, compute_regions: bool):
        """
        Fetches histograms for a subset of features from the clients.

        Args:
            feature_subset (List): A list of features for which histograms are requested.
            compute_regions (bool): Whether to compute regions during histogram calculation.

        Returns:
            dict: A dictionary of histograms aggregated across clients.

        Raises:
            Exception: If communication with any client fails.
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

        # Aggregate histograms across clients.
        histogram = {k: np.sum(list(v.values()), axis=0) for k, v in histogram.items()}
        return histogram

    def evaluate(self):
        """
        Evaluates the performance of the federated XGBoost model across all clients.

        Returns:
            dict: A dictionary containing the evaluation results aggregated across clients.

        Raises:
            Exception: If communication with any client fails.
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
