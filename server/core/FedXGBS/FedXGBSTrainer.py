import numpy as np
from ..Models.fedxgb import XGBoostTree
from typing import Dict


class FedXGBSTrainer:
    """
    Trainer for the federated XGBoost model, responsible for managing the feature splits,
    boosting (model updates), and feature sampling in a distributed setup.

    Attributes:
        feature_importance (Dict[int, float]): The importance of each feature.
        binary (Dict[int, np.ndarray]): Binary feature data for client-side calculations.
        quantiles (Dict[int, np.ndarray]): Quantiles for each feature.
        base_y (float): The base label value.
        self_calculate (bool): Flag to determine whether to calculate feature importance locally.
        feature_splits (Dict[int, List[float]]): The split values for each feature.
        splits_per_feature (Dict[int, int]): Number of splits for each feature.
        excluded_features (List[int]): Features that are excluded from the splits due to low importance.
    """

    def __init__(
            self,
            feature_importance: Dict[int, float],
            binary: Dict[int, np.ndarray],
            quantiles: Dict[int, np.ndarray],
            base_y: float,
            self_calculate: bool = False
        ):
        """
        Initializes the FedXGBSTrainer instance.

        Args:
            feature_importance (Dict[int, float]): The initial importance scores of features.
            binary (Dict[int, np.ndarray]): Binary feature data.
            quantiles (Dict[int, np.ndarray]): Quantiles for each feature.
            base_y (float): The base value for the target variable.
            self_calculate (bool, optional): Whether to calculate feature importance locally. Defaults to False.
        """
        self.feature_importance = feature_importance
        self.binary = binary
        self.quantiles = quantiles
        self.base_y = base_y
        self.self_calculate = self_calculate

    def make_splits(self, avg_splits: int = 2):
        """
        Creates splits for each feature based on the average number of splits and the feature's importance.

        Args:
            avg_splits (int, optional): The average number of splits per feature. Defaults to 2.
        """
        # Assign splits to features based on their importance
        self.splits_per_feature = self.assign_splits(avg_splits)

        # Remove excluded features from the importance list
        for feature in self.excluded_features:
            self.feature_importance.pop(feature)

        # Create the splits for each feature
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
                # For binary features, we use the mean as the split
                feature_splits[feature] = [float(np.mean(self.binary[feature]))]
        
        self.feature_splits = feature_splits

    def assign_splits(self, avg_splits: int):
        """
        Assigns the number of splits to each feature based on its importance.

        Args:
            avg_splits (int): The average number of splits per feature.

        Returns:
            Dict[int, int]: A dictionary mapping each feature to the number of splits assigned.
        """
        features = len(self.feature_importance)
        total_importance = sum(self.feature_importance.values())

        # Check for zero total importance to avoid division by zero
        if total_importance == 0:
            raise ValueError("Total feature importance is zero. Cannot assign splits.")

        total_splits = avg_splits * features
        splits_per_feature = {}

        for feature in self.feature_importance:
            if (feature not in self.binary) or (round(total_splits * self.feature_importance[feature] / total_importance) == 0):
                splits_per_feature[feature] = round(total_splits * self.feature_importance[feature] / total_importance)
            else:
                splits_per_feature[feature] = 1

        # Adjust splits to ensure the total adds up to total_splits
        allocated_splits = sum(splits_per_feature.values())
        discrepancy = total_splits - allocated_splits

        # Distribute discrepancy to non-binary features
        if discrepancy != 0:
            non_binary_features = [f for f in self.feature_importance if f not in self.binary]
            adjustments = np.sign(discrepancy) * np.ones(abs(discrepancy), dtype=int)

            for i, feature in enumerate(non_binary_features[:len(adjustments)]):
                splits_per_feature[feature] += adjustments[i]

        # Exclude features with zero splits
        self.excluded_features = [feature for feature, splits in splits_per_feature.items() if splits == 0]
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
        """
        Sets the parameters for the boosting process.

        Args:
            min_child_weight (float, optional): Minimum child weight. Defaults to 1.0.
            depth (int, optional): Maximum depth of the trees. Defaults to 5.
            min_leaf (int, optional): Minimum number of leaves. Defaults to 5.
            lambda_ (float, optional): Lambda regularization parameter. Defaults to 1.5.
            gamma (float, optional): Gamma parameter for tree pruning. Defaults to 1.
            features_per_booster (int, optional): Number of features to use per booster. Defaults to 10.
            importance_rounds (int, optional): Number of rounds for calculating feature importance. Defaults to 20.
        """
        self.min_child_weight = min_child_weight
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.features_per_booster = features_per_booster
        self.importance_rounds = importance_rounds

    def boost(self, round_num, client_manager, feature_importance: Dict[int, float]):
        """
        Performs a boosting step, training a tree based on the client's data.

        Args:
            round_num (int): The current round of boosting.
            client_manager: The manager responsible for fetching the data and histograms.
            feature_importance (Dict[int, float]): Feature importance values used in boosting.

        Returns:
            XGBoostTree: The trained tree (XGBoostTree) for this round of boosting.
        """
        sampling = self.features_per_booster < len(self.feature_importance)
        features_subset = FedXGBSTrainer._sample_n_features(self.features_per_booster, self.feature_importance) if sampling else list(self.feature_importance.keys())

        feature_splits = {feature: self.feature_splits[feature] for feature in features_subset}
        compute_regions = (round_num == 0 or sampling)

        # Get histograms from the client
        global_histogram = client_manager.get_histograms(features_subset, compute_regions)

        # Train the tree using the histograms
        boosting_tree = XGBoostTree().hist_fit(
            histogram=global_histogram,
            feature_splits=feature_splits,
            min_leaf=self.min_leaf,
            min_child_weight=self.min_child_weight,
            depth=self.depth,
            lambda_=self.lambda_,
            gamma=self.gamma,
            feature_importance=feature_importance
        )
            
        return boosting_tree

    @staticmethod
    def _sample_n_features(n: int, feature_importance: Dict[int, float]):
        probabilities = np.array(list(feature_importance.values()), dtype=np.float64)
        probabilities /= probabilities.sum()

        return np.random.choice(list(feature_importance.keys()), n, p=probabilities, replace=False).tolist()
