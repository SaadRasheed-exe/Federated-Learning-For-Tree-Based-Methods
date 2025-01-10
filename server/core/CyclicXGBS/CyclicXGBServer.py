from ..Utility import BaseServer
from .CyclicXGBClientManager import CyclicXGBClientManager
from xgboost import XGBClassifier
from tqdm import tqdm
from typing import Dict


class CyclicXGBServer(BaseServer):

    def __init__(self, clients_json_path):
        super().__init__(clients_json_path)
        self.client_manager = CyclicXGBClientManager(
            self.clients,
            self.encryption_manager
        )
        self.client_manager.init_encryption('cyclic')

    def fit(
            self, 
            model: XGBClassifier, 
            weightage: Dict[str, int]
        ) -> XGBClassifier:

        for client, num_rounds in tqdm(weightage.items()):
            updated_model = self.client_manager.train_client(client, model, num_rounds)
            model = updated_model

        return model

    def evaluate(self, model: XGBClassifier):
        scores = {}
        for client in self.clients:
            client_scores = self.client_manager.evaluate_client(client, model)
            for key, value in client_scores.items():
                if key in scores:
                    scores[key] += value
                else:
                    scores[key] = value
        
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
        for key, value in params.items():
            if key in ['max_depth']:
                params[key] = int(value)
            elif key in ['learning_rate']:
                params[key] = float(value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        
        return params