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
    def train_clients(self, model: Any, time: datetime):
        """
        Train all active clients using their local data.
        Args:
            time (str): The current time for logging purposes.
        """
        serialized_model = pickle.dumps(model).hex()
        serialized_time = time.strftime('%Y-%m-%d %H:%M:%S')
        data = {'time': serialized_time, 'model': serialized_model}

        client_models = {}
        # decoded_time = time.strftime('%Y-%m-%d %H:%M:%S')
        responses = self._execute_in_threads(4, self._communicate, 'agg/train', data, serialize=False)
        for client_id, response in responses.items():
            if response:
                client_models[client_id] = pickle.loads(bytes.fromhex(response.get('model')))
        return client_models
    
    def evaluate_clients(self, model: MajorityVotingEnsemble):
        """
        Evaluate all active clients using the provided model.
        Args:
            model (Any): The model to evaluate.
        Returns:
            scores (dict): A dictionary of evaluation scores for each client.
        """
        serialized_model = pickle.dumps(model.serialize()).hex()
        data = {'model': serialized_model}
        scores = self._execute_in_threads(4, self._communicate, 'agg/evaluate', data, serialize=False)
        return scores