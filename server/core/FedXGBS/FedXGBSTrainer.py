import numpy as np
from ..Models.fedxgb import XGBoostTree
from typing import Dict


class FedXGBSTrainer:

    def __init__(
            self,
            feature_importance: Dict[int, float],
            binary: Dict[int, np.ndarray],
            quantiles: Dict[int, np.ndarray],
            base_y: float,
            self_calculate: bool = False
        ):
        self.feature_importance = feature_importance
        self.binary = binary
        self.quantiles = quantiles
        self.base_y = base_y
        self.self_calculate = self_calculate

    def make_splits(self, avg_splits: int = 2):
        '''
        Assigns splits to each feature based on the average number of splits per feature and feature importance.
        '''
        self.splits_per_feature = self.assign_splits(avg_splits)

        for feature in self.excluded_features:
            self.feature_importance.pop(feature)

        feature_splits = {}
        for feature, splits in self.splits_per_feature.items():
            feature_values = self.quantiles[feature]
            if feature not in self.binary:
                feature_splits[feature] = np.unique(
                    np.quantile(
                        feature_values, 
                        q=np.linspace(0, 1, splits + 2)[1:-1]
                    )
                ).tolist()
            else:
                feature_splits[feature] = [float(np.mean(self.binary[feature]))]
        
        self.feature_splits = feature_splits
    
    def assign_splits(self, avg_splits):
        '''
        Assigns splits to each feature based on the average number of splits per feature and feature importance.
        '''
        features = len(self.feature_importance)
        total_importance = sum(self.feature_importance.values())

        # Check for zero total importance to avoid division by zero
        if total_importance == 0:
            raise ValueError("Total feature importance is zero. Cannot assign splits.")

        total_splits = avg_splits * features

        splits_per_feature = {}
        for feature in self.feature_importance:
            if (feature not in self.binary) or (round(total_splits * self.feature_importance[feature] / total_importance)==0):
                splits_per_feature[feature] = round(total_splits * self.feature_importance[feature] / total_importance)
            else:
                splits_per_feature[feature] = 1

        # Initial allocation of splits
        # splits_per_feature = {
        #     feature: round(total_splits * self.feature_importance[feature] / total_importance)
        #     if (feature not in self.binary) or (round(total_splits * self.feature_importance[feature] / total_importance)==0) else 1
        #     for feature in self.feature_importance
        # }

        # Adjust splits to ensure the total adds up to total_splits
        allocated_splits = sum(splits_per_feature.values())
        discrepancy = total_splits - allocated_splits

        # Adjust by distributing the discrepancy to non-binary features
        if discrepancy != 0:
            non_binary_features = [f for f in self.feature_importance if f not in self.binary]
            adjustments = np.sign(discrepancy) * np.ones(abs(discrepancy), dtype=int)

            for i, feature in enumerate(non_binary_features[:len(adjustments)]):
                splits_per_feature[feature] += adjustments[i]

        self.excluded_features = []

        for feature in splits_per_feature:
            if splits_per_feature[feature] == 0:
                self.excluded_features.append(feature)

        # remove excluded features
        for feature in self.excluded_features:
            splits_per_feature.pop(feature)

        return splits_per_feature

    def set_params(
            self,
            min_child_weight: float = 1.0,
            depth: int = 5,
            min_leaf: int = 5,
            lambda_: float = 1.5,
            gamma: float = 1,
            features_per_booster: int = 10,
            importance_rounds: int = 20
        ):
        '''
        Sets the parameters for the tree model.
        '''
        self.min_child_weight = min_child_weight
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.features_per_booster = features_per_booster
        self.importance_rounds = importance_rounds

    def boost(self, round_num, client_manager, feature_importance: Dict[int, float]):
        '''
        Boosts the model by training a new tree and updating the feature importance.
        '''
        sampling = self.features_per_booster < len(self.feature_importance)
        if sampling:
            features_subset = self.sample_n_features(self.features_per_booster, self.feature_importance)
        else:
            features_subset = list(self.feature_importance.keys())

        feature_splits = {feature: self.feature_splits[feature] for feature in features_subset}
        compute_regions = (round_num == 0 or sampling)

        global_histogram = client_manager.get_histograms(features_subset, compute_regions)

        boosting_tree = XGBoostTree().hist_fit(
            histogram=global_histogram,
            feature_splits=feature_splits,
            min_leaf=self.min_leaf,
            min_child_weight=self.min_child_weight,
            depth = self.depth,
            lambda_ = self.lambda_,
            gamma = self.gamma,
            feature_importance=feature_importance
        )
            
        return boosting_tree

    @staticmethod
    def sample_n_features(n: int, feature_importance: Dict[int, float]):
        '''
        Samples n features from the feature importance distribution.
        '''
        # Convert feature importance values to a numpy array and cast to float64
        probabilities = np.array(list(feature_importance.values()), dtype=np.float64)

        # Normalize probabilities to ensure they sum exactly to 1
        probabilities /= probabilities.sum()

        return np.random.choice(list(feature_importance.keys()), n, p=list(probabilities), replace=False).tolist()
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
