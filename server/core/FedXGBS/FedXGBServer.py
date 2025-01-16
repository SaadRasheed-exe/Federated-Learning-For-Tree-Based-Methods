from tqdm import tqdm
from .FedXGBClientManager import FedXGBClientManager
from typing import Literal
from .FedXGBSTrainer import FedXGBSTrainer
from ..Models.fedxgb import FedXGBoostEnsemble
from ..Utility import BaseServer

class FedXGBServer(BaseServer):
    """
    A server class for managing and training a federated XGBoost model in a distributed setup.

    This class handles the communication with clients, initialization of training parameters,
    and orchestration of federated model training using XGBoost.

    Attributes:
        client_manager (FedXGBClientManager): Manages client-side communication and data retrieval.
        features (List[str]): List of feature names in the dataset.
        num_features (int): Number of features in the dataset.
        estimators (List): List of estimators (trees) used in the federated model.
        avg_splits (int): Average number of splits for feature binning.
        self_calculate (bool): Whether to calculate feature importance locally or rely on client-side values.
        global_feature_splits (dict): Stores the global feature splits.
        trainer (FedXGBSTrainer): Trainer responsible for training the XGBoost models.
    """

    def __init__(self, clients_json_path: str):
        """
        Initializes the FedXGBServer instance.

        Args:
            clients_json_path (str): Path to the JSON configuration containing client details.
        """
        super().__init__(clients_json_path)
        self.client_manager = FedXGBClientManager(self.clients)

        # Fetch feature names and initialize clients.
        self.features = self.client_manager.fetch_feature_names()
        self.num_features = len(self.features)
        self.client_manager.init_clients(self.features)
        self.client_manager.init_masks()

        # Initialize list of estimators (trees).
        self.estimators = []

    def initialize(self, avg_splits: int = 2, importance_method: Literal['gain', 'xgb'] = 'gain'):
        """
        Initializes the federated learning setup by fetching data from clients, 
        calculating feature importance, and setting up the trainer.

        Args:
            avg_splits (int, optional): The average number of splits for feature binning. Defaults to 2.
            importance_method (str, optional): Method for calculating feature importance ('gain' or 'xgb'). Defaults to 'gain'.
        """
        self.avg_splits = avg_splits
        self.self_calculate = (importance_method == 'gain')
        self.global_feature_splits = {}

        # Fetch necessary data from clients.
        binary = self.client_manager.get_binary(self.num_features)
        ones, n_samples = self.client_manager.get_y()
        base_y = ones / n_samples
        quantiles = self.client_manager.get_quantiles()

        # Initialize feature importance either locally or from client data.
        if self.self_calculate:
            feature_importance = {feature: 1 for feature in range(self.num_features)}  # equal importance to start
        else:
            feature_importance = self.client_manager.get_feature_importance()

        # Initialize the trainer for the federated XGBoost system.
        self.trainer = FedXGBSTrainer(
            feature_importance,
            binary,
            quantiles,
            base_y,
            self.self_calculate
        )

        # Perform initial splitting of features.
        self.trainer.make_splits(avg_splits)
        self.client_manager.set_base_y(base_y)
        self.client_manager.set_feature_splits(self.trainer.feature_splits)

    def fit_generator(
        self,
        resume=False, 
        min_child_weight=1.0, 
        depth=5, 
        min_leaf=5,
        learning_rate=0.3, 
        boosting_rounds=5, 
        lambda_=1.5, 
        gamma=1,
        features_per_booster=10, 
        importance_rounds=20
    ):
        """
        Generates the federated model training process. This function performs 
        multiple rounds of boosting, updates the estimators, and calculates feature importance.

        Args:
            resume (bool, optional): Whether to resume training from existing estimators. Defaults to False.
            min_child_weight (float, optional): Minimum child weight for boosting. Defaults to 1.0.
            depth (int, optional): Maximum depth of trees. Defaults to 5.
            min_leaf (int, optional): Minimum number of leaves in the tree. Defaults to 5.
            learning_rate (float, optional): Learning rate for boosting. Defaults to 0.3.
            boosting_rounds (int, optional): Number of boosting rounds. Defaults to 5.
            lambda_ (float, optional): Lambda regularization parameter. Defaults to 1.5.
            gamma (float, optional): Gamma parameter for pruning trees. Defaults to 1.
            features_per_booster (int, optional): Number of features per booster. Defaults to 10.
            importance_rounds (int, optional): Number of rounds to calculate feature importance. Defaults to 20.

        Yields:
            int: The current round number of boosting during training.

        Returns:
            FedXGBoostEnsemble: The trained ensemble of XGBoost models.
        """
        self.client_manager.set_learning_rate(learning_rate)

        if not resume:
            self.estimators = []
            self.client_manager.set_estimators([])
        elif len(self.estimators) == 0:
            raise ValueError('No estimators to resume training from')

        # Set training parameters.
        self.trainer.set_params(
            min_child_weight=min_child_weight,
            depth=depth,
            min_leaf=min_leaf,
            lambda_=lambda_,
            gamma=gamma,
            features_per_booster=features_per_booster,
            importance_rounds=importance_rounds
        )

        # Initialize feature importance.
        if self.self_calculate:
            feature_importance = {feature: 0 for feature in self.trainer.feature_importance}
        else:
            feature_importance = None

        # Perform boosting for the specified number of rounds.
        for round_num in range(boosting_rounds):
            new_tree = self.trainer.boost(
                round_num, 
                self.client_manager,
                feature_importance
            )
            self.client_manager.add_estimator(new_tree.to_dict())
            self.estimators.append(new_tree)

            # Update feature importance if required.
            if self.self_calculate and round_num + 1 == importance_rounds:
                min_importance = min(feature_importance.values())
                feature_importance = {k: v - min_importance for k, v in feature_importance.items()}
                self.trainer.feature_importance = feature_importance
                self.trainer.make_splits(self.avg_splits)
                feature_importance = None
                self.client_manager.set_feature_splits(self.trainer.feature_splits)

            yield round_num

        # Return the trained ensemble of models.
        return FedXGBoostEnsemble(
            self.estimators,
            self.trainer.base_y,
            learning_rate,
            self.features,
            self.trainer.feature_importance
        )

    def fit(
        self,
        resume=False, 
        min_child_weight=1.0, 
        depth=5, 
        min_leaf=5,
        learning_rate=0.3, 
        boosting_rounds=5, 
        lambda_=1.5, 
        gamma=1,
        features_per_booster=10, 
        importance_rounds=20
    ):
        """
        Initiates the training process by calling the fit_generator method 
        and processing the results until completion.

        Args:
            resume (bool, optional): Whether to resume training. Defaults to False.
            min_child_weight (float, optional): Minimum child weight for boosting. Defaults to 1.0.
            depth (int, optional): Maximum depth of trees. Defaults to 5.
            min_leaf (int, optional): Minimum number of leaves in the tree. Defaults to 5.
            learning_rate (float, optional): Learning rate for boosting. Defaults to 0.3.
            boosting_rounds (int, optional): Number of boosting rounds. Defaults to 5.
            lambda_ (float, optional): Lambda regularization parameter. Defaults to 1.5.
            gamma (float, optional): Gamma parameter for pruning trees. Defaults to 1.
            features_per_booster (int, optional): Number of features per booster. Defaults to 10.
            importance_rounds (int, optional): Number of rounds for calculating feature importance. Defaults to 20.

        Returns:
            FedXGBoostEnsemble: The trained federated XGBoost ensemble model.
        """
        fit_gen = self.fit_generator(
                resume=resume, 
                min_child_weight=min_child_weight, 
                depth=depth, 
                min_leaf=min_leaf,
                learning_rate=learning_rate, 
                boosting_rounds=boosting_rounds, 
                lambda_=lambda_, 
                gamma=gamma,
                features_per_booster=features_per_booster, 
                importance_rounds=importance_rounds
            )
        try:
            while True:
                next(fit_gen)
        except StopIteration as e:
            return e.value

    def evaluate(self, *args):
        """
        Evaluates the performance of the federated model by calculating metrics like accuracy, precision, 
        recall, and F1-score based on client-side predictions.

        Returns:
            dict: A dictionary containing evaluation metrics (accuracy, precision, recall, F1-score).
        """
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
            'f1': f1
        }

    def parse_params(self, params):
        """
        Parses the hyperparameters for model training from a dictionary and converts them to appropriate types.

        Args:
            params (dict): Dictionary containing parameter names and their corresponding values.

        Returns:
            dict: Dictionary with parsed parameter values.
        
        Raises:
            ValueError: If a parameter name is invalid.
        """
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
