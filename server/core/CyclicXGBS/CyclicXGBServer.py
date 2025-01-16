from ..Utility import BaseServer
from .CyclicXGBClientManager import CyclicXGBClientManager
from xgboost import XGBClassifier
from tqdm import tqdm
from typing import Dict


class CyclicXGBServer(BaseServer):
    """
    A server-side class that manages the training and evaluation of XGBoost models
    in a cyclic federated learning setup. The server coordinates communication with
    clients, aggregates results, and manages model training and evaluation processes.
    """

    def __init__(self, clients_json_path):
        """
        Initializes the server by loading the clients and setting up the client manager.

        Args:
            clients_json_path (str): Path to the configuration file containing client details.
        """
        super().__init__(clients_json_path)
        # Initialize the client manager for handling client-side operations.
        self.client_manager = CyclicXGBClientManager(self.clients)

    def fit(self, model: XGBClassifier, weightage: Dict[str, int]) -> XGBClassifier:
        """
        Trains the provided XGBoost model on each client for a specified number of rounds.

        Args:
            model (XGBClassifier): The XGBoost model to be trained.
            weightage (Dict[str, int]): A dictionary mapping client IDs to the number of training rounds.

        Returns:
            XGBClassifier: The updated XGBoost model after training on all clients.
        """
        # Iterate over each client and train the model for the specified number of rounds.
        for client, num_rounds in tqdm(weightage.items()):
            # Train the model on the current client.
            updated_model = self.client_manager.train_client(client, model, num_rounds)
            # Update the model with the client's trained version.
            model = updated_model

        return model

    def evaluate(self, model: XGBClassifier):
        """
        Evaluates the provided XGBoost model on all clients and computes overall performance metrics.

        Args:
            model (XGBClassifier): The XGBoost model to be evaluated.

        Returns:
            dict: A dictionary containing evaluation metrics such as precision, recall, f1 score, and accuracy.
        """
        scores = {}
        # Evaluate the model on each client and aggregate the results.
        for client in self.clients:
            client_scores = self.client_manager.evaluate_client(client, model)
            for key, value in client_scores.items():
                # Accumulate the scores from each client.
                if key in scores:
                    scores[key] += value
                else:
                    scores[key] = value
        
        # Calculate performance metrics based on the aggregated scores.
        tp = scores['tp']
        fp = scores['fp']
        fn = scores['fn']
        tn = scores['tn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def parse_params(self, params):
        """
        Parses and converts hyperparameters from string format to appropriate types.

        Args:
            params (dict): A dictionary of hyperparameters as strings.

        Returns:
            dict: A dictionary of parsed hyperparameters with correct types.
        
        Raises:
            ValueError: If an invalid parameter is encountered.
        """
        for key, value in params.items():
            # Convert specific hyperparameters to the correct type.
            if key in ['max_depth']:
                params[key] = int(value)
            elif key in ['learning_rate']:
                params[key] = float(value)
            else:
                # Raise an error if an invalid parameter is encountered.
                raise ValueError(f"Invalid parameter: {key}")
        
        return params
