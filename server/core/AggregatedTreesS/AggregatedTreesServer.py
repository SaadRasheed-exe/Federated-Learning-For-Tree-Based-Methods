from typing import Dict, List
from datetime import datetime
from ..Models.agg import MajorityVotingEnsemble
from ..Utility import BaseServer
from .AggregatedTreesClientManager import AggregatedTreesClientManager

class AggregatedTreesServer(BaseServer):
    """
    Represents a server for managing aggregated tree models in a federated learning setup.
    It coordinates client training, model aggregation, and evaluation.
    """

    def __init__(self, clients_json_path: str):
        """
        Initializes the AggregatedTreesServer.

        Args:
            clients_json_path (str): Path to the JSON file containing client configuration.
        """
        super().__init__(clients_json_path)
        self.client_manager = AggregatedTreesClientManager(self.clients)

    def fit(
            self,
            model,
            weightage: Dict[str, float] = None,
            participants: List = None,
        ):
        """
        Trains a model using the specified clients and aggregates the results.

        Args:
            model: The base model to be trained.
            weightage (Dict[str, float], optional): Weights for each participant in the ensemble. Defaults to None.
            participants (List, optional): List of clients to participate in training. Defaults to all clients.

        Returns:
            MajorityVotingEnsemble: Combined model created using majority voting.
        """
        if participants is None:
            participants = list(self.clients.keys())
        else:
            self.client_manager.active_clients = participants

        time = datetime.now()  # Record the current time for training.
        if weightage:
            model_weightage = {k: weightage[k] for k in participants}  # Filter weights for active participants.

        feature_names = self.client_manager.get_feature_names()
        client_models = self.client_manager.train_clients(model, time, feature_names)
        combined_model = MajorityVotingEnsemble(client_models, model_weightage) if weightage else MajorityVotingEnsemble(client_models)
        return combined_model

    def evaluate(self, model: MajorityVotingEnsemble):
        """
        Evaluates a combined model across all clients.

        Args:
            model (MajorityVotingEnsemble): The aggregated model to evaluate.

        Returns:
            dict: A dictionary containing precision, recall, F1 score, and accuracy.
        """
        client_scores = self.client_manager.evaluate_clients(model)
        
        # Initialize metrics.
        tp = fp = fn = tn = 0
        for _, scores in client_scores.items():
            tp += scores['tp']
            fp += scores['fp']
            fn += scores['fn']
            tn += scores['tn']
        
        # Calculate performance metrics.
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
    
    def parse_params(self, params: Dict[str, str]):
        """
        Parses and validates model parameters provided as strings, converting them to appropriate types.

        Args:
            params (Dict[str, str]): Dictionary of model parameters as strings.

        Returns:
            Dict: Parsed parameters with appropriate types.

        Raises:
            ValueError: If an invalid parameter is encountered.
        """
        for param in params:
            if param == 'learning_rate':
                params[param] = float(params[param])  # Convert learning rate to float.
            elif param == 'n_estimators':
                params[param] = int(params[param])  # Convert number of estimators to integer.
            elif param == 'max_depth':
                params[param] = int(params[param])  # Convert max depth to integer.
            elif param == 'boosting_type':
                params[param] = params[param].lower()  # Ensure boosting type is lowercase.
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return params
