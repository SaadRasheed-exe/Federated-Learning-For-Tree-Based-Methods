from tqdm import tqdm
from .FedXGBClientManager import FedXGBClientManager
from typing import Literal
from .FedXGBSTrainer import FedXGBSTrainer
from ..Models.fedxgb import FedXGBoostEnsemble
from ..Utility import BaseServer


class FedXGBServer(BaseServer):

    def __init__(self, clients_json_path: str):
        super().__init__(clients_json_path)
        self.client_manager = FedXGBClientManager(self.clients, self.encryption_manager)

        self.client_manager.init_encryption('fedxgb')
        self.features = self.client_manager.fetch_feature_names()
        self.num_features = len(self.features)
        self.client_manager.init_clients(self.features)
        self.client_manager.init_masks()
        self.estimators = []

    def initialize(self, avg_splits: int = 2, importance_method: Literal['gain', 'xgb'] = 'gain'):
        self.avg_splits = avg_splits
        self.self_calculate = (importance_method == 'gain')
        self.global_feature_splits = {}
        
        binary = self.client_manager.get_binary(self.num_features)
        ones, n_samples = self.client_manager.get_y()
        base_y = ones / n_samples
        quantiles = self.client_manager.get_quantiles()
        
        if self.self_calculate:
            feature_importance = {feature: 1 for feature in range(self.num_features)} # initialize equal feature importance
        else:
            feature_importance = self.client_manager.get_feature_importance()
        
        self.trainer = FedXGBSTrainer(
            feature_importance,
            binary,
            quantiles,
            base_y,
            self.self_calculate
        )

        self.trainer.make_splits(avg_splits)
        self.client_manager.set_base_y(base_y)
        self.client_manager.set_feature_splits(self.trainer.feature_splits)
    

    def fit(
        self,
        resume = False, 
        min_child_weight = 1.0, 
        depth = 5, 
        min_leaf = 5,
        learning_rate = 0.3, 
        boosting_rounds = 5, 
        lambda_ = 1.5, 
        gamma = 1,
        features_per_booster = 10, 
        importance_rounds = 20,
        verbose=True
    ):

        self.client_manager.set_learning_rate(learning_rate)
        if not resume:
            self.estimators = []
            self.client_manager.set_estimators([])
        elif len(self.estimators) == 0:
            raise ValueError('No estimators to resume training from')
        
        self.trainer.set_params(
            min_child_weight=min_child_weight,
            depth=depth,
            min_leaf=min_leaf,
            lambda_=lambda_,
            gamma=gamma,
            features_per_booster=features_per_booster,
            importance_rounds=importance_rounds
        )

        if self.self_calculate:
            feature_importance = {feature: 0 for feature in self.trainer.feature_importance}
        else:
            feature_importance = None

        iterator = tqdm(range(boosting_rounds)) if verbose else range(boosting_rounds)

        for round_num in iterator:
            new_tree = self.trainer.boost(
                round_num, 
                self.client_manager,
                feature_importance
            )
            self.client_manager.add_estimator(new_tree.to_dict())
            self.estimators.append(new_tree)

            if self.self_calculate and round_num + 1 == importance_rounds:
                min_importance = min(feature_importance.values())
                feature_importance = {k: v - min_importance for k, v in feature_importance.items()}
                
                self.trainer.feature_importance = feature_importance
                self.trainer.make_splits(self.avg_splits)
                feature_importance = None
                self.client_manager.set_feature_splits(self.trainer.feature_splits)
        
        return FedXGBoostEnsemble(
            self.estimators,
            self.trainer.base_y,
            learning_rate,
            self.features
        )

    def evaluate(self):
        counts = self.client_manager.evaluate()
        tp = counts['tp']
        tn = counts['tn']
        fp = counts['fp']
        fn = counts['fn']

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def parse_params(self, params):
        for param in params:
            if param in [
                "avg_splits", "features_per_booster", 
                "importance_rounds", "boosting_rounds",
                "depth", "min_leaf"]:
                params[param] = int(params[param])
            elif param in [
                "min_child_weight", "learning_rate", 
                "lambda_", "gamma"]:
                params[param] = float(params[param])
            else:
                raise ValueError(f"Invalid parameter {param}")
        
        return params
