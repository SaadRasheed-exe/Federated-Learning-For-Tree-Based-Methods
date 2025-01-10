from typing import Dict, List
from datetime import datetime
from ..Models.agg import MajorityVotingEnsemble
from ..Utility import BaseServer
from .AggregatedTreesClientManager import AggregatedTreesClientManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class AggregatedTreesServer(BaseServer):

    def __init__(self, clients_json_path: str):
        super().__init__(clients_json_path)
        self.client_manager = AggregatedTreesClientManager(
            self.clients,
            self.encryption_manager
        )
        self.client_manager.init_encryption('agg')
    
    def send_code_dir(self, code_dir: str):
        self.client_manager.send_code_dir(code_dir)

    def fit(
            self,
            model,
            weightage: Dict[str, float] = None,
            participants: List = None,
        ):

        if participants is None:
            participants = list(self.clients.keys())
        else:
            self.client_manager.active_clients = participants

        time = datetime.now()    
        if weightage:
            model_weightage = {k: weightage[k] for k in participants}
        
        client_models = self.client_manager.train_clients(model, time)
        combined_model = MajorityVotingEnsemble(client_models, model_weightage) if weightage else MajorityVotingEnsemble(client_models)
        return combined_model
    
    def evaluate(self, model: MajorityVotingEnsemble):
        client_scores = self.client_manager.evaluate_clients(model)
        
        tp = fp = fn = tn = 0
        for _, scores in client_scores.items():
            tp += scores['tp']
            fp += scores['fp']
            fn += scores['fn']
            tn += scores['tn']
        
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
        for param in params:
            if param == 'learning_rate':
                params[param] = float(params[param])
            elif param == 'n_estimators':
                params[param] = int(params[param])
            elif param == 'max_depth':
                params[param] = int(params[param])
            elif param == 'boosting_type':
                params[param] = params[param].lower()
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return params
