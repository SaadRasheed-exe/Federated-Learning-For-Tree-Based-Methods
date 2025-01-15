from ..Utility import BaseClientManager
from xgboost import XGBClassifier

class CyclicXGBClientManager(BaseClientManager):
    """
    A class that manages client-side operations for training and evaluating XGBoost models
    in a cyclic federated learning setup.
    """

    def train_client(self, client_id, model: XGBClassifier, num_rounds: int) -> XGBClassifier:
        """
        Trains an XGBoost model on a specific client for a specified number of rounds.

        Args:
            client_id (str): The ID of the client where the model will be trained.
            model (XGBClassifier): The XGBoost model to be trained.
            num_rounds (int): The number of rounds for training the model.

        Returns:
            XGBClassifier: The updated XGBoost model after training.
        """
        # Prepare data to send to the client for training.
        data = {'model': model, 'num_rounds': num_rounds}
        
        # Communicate with the client to perform the training.
        response = self._communicate(client_id, 'cyclic/train', data)
        
        # Extract the updated model from the response.
        updated_model = response.get('model')
        return updated_model
    
    def evaluate_client(self, client_id: str, model: XGBClassifier):
        """
        Evaluates an XGBoost model on a specific client and returns the evaluation scores.

        Args:
            client_id (str): The ID of the client where the model will be evaluated.
            model (XGBClassifier): The XGBoost model to be evaluated.

        Returns:
            dict: A dictionary containing evaluation scores for the model.
        """
        # Prepare data to send to the client for evaluation.
        data = {'model': model}
        
        # Communicate with the client to perform the evaluation.
        scores = self._communicate(client_id, 'cyclic/evaluate', data)
        return scores
