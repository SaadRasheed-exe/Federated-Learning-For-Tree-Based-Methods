from typing import Any
import pickle
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from ..Models.agg import MajorityVotingEnsemble
from ..Utility import BaseClientManager

class AggregatedTreesClientManager(BaseClientManager):
    """
    Manages the client-side operations for aggregated tree models in a federated learning setup.
    This class handles communication with clients, training models, and evaluating results.
    """

    def get_feature_names(self):
        """
        Communicates with each active client to retrieve the feature names.

        Returns:
            list: A list of feature names from the first active client.

        Raises:
            ValueError: If feature names are not found.
        """
        for client_id in self.active_clients:
            feature_names = self._communicate(client_id, 'agg/get_feature_names', serialize=False)
            if feature_names is not None:
                return feature_names.get('feature_names')
        # If no feature names are found, raise an error.
        raise ValueError("Feature names not found in any active client.")
    
    def train_clients(self, model: Any, time: datetime, feature_names: list):
        """
        Trains the provided model on all active clients and returns the client-specific models.

        Args:
            model (Any): The model to be trained (e.g., XGBClassifier, RandomForestClassifier).
            time (datetime): The timestamp of the training session.
            feature_names (list): The list of feature names for training.

        Returns:
            dict: A dictionary of client IDs and their respective trained models.
        """
        # Serialize the model and time to send to clients.
        serialized_model = pickle.dumps(model).hex()
        serialized_time = time.strftime('%Y-%m-%d %H:%M:%S')  # Convert time to string format.
        data = {'time': serialized_time, 'model': serialized_model, 'feature_names': feature_names}

        client_models = {}  # Dictionary to store the models from each client.
        
        # Execute training in parallel across clients using threads.
        responses = self._execute_in_threads(4, self._communicate, 'agg/train', data, serialize=False)
        for client_id, response in responses.items():
            if response:
                # Deserialize and store the model received from the client.
                client_models[client_id] = pickle.loads(bytes.fromhex(response.get('model')))
        
        return client_models
    
    def evaluate_clients(self, model: MajorityVotingEnsemble):
        """
        Evaluates the provided model across all active clients and returns evaluation scores.

        Args:
            model (MajorityVotingEnsemble): The aggregated model to be evaluated.

        Returns:
            dict: A dictionary of client IDs and their evaluation scores.
        """
        data = {'model': model}

        # Execute evaluation in parallel across clients using threads.
        scores = self._execute_in_threads(4, self._communicate, 'agg/evaluate', data)
        
        return scores
