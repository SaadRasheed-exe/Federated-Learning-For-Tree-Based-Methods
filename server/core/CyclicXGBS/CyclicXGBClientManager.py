from ..Utility import BaseClientManager
import pickle
from xgboost import XGBClassifier

class CyclicXGBClientManager(BaseClientManager):
    
    def train_client(self, client_id, model: XGBClassifier, num_rounds: int) -> XGBClassifier:
        """
        Train a single client using its local data.
        Args:
            client_id (str): The ID of the client to train.
            model (Any): The model to train.
        Returns:
            model (Any): The updated model.
        """
        data = {'model': model, 'num_rounds': num_rounds}
        response = self._communicate(client_id, 'cyclic/train', data)
        updated_model = response.get('model')
        return updated_model
    
    def evaluate_client(self, client_id: str, model: XGBClassifier):
        """
        Evaluate a single client using the provided model.
        Args:
            client_id (str): The ID of the client to evaluate.
            model (Any): The model to evaluate.
        Returns:
            scores (dict): A dictionary of evaluation scores.
        """
        data = {'model': model}
        scores = self._communicate(client_id, 'cyclic/evaluate', data)
        return scores
    
    def send_config(self, client_id, config):
        """
        Send the configuration to a client.
        Args:
            client_id (str): The ID of the client to send the configuration to.
            config (ConfigParser): The configuration to send.
        """
        data = {'config': config}
        self._communicate(client_id, 'cyclic/config', data)