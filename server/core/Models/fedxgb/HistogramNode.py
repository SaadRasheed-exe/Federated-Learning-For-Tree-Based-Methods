import numpy as np
from typing import Dict, List, Optional, Union

class HistogramNode:
    """
    A class representing a node in a histogram-based decision tree. This tree is used for gradient boosting models
    where nodes split based on feature values in order to minimize the loss function. The node uses histograms to
    efficiently calculate gradients and hessians for tree construction.

    Attributes:
    - NEGATIVE_INFINITY (float): A constant representing negative infinity for initialization of scores.
    - depth (int): The maximum depth of the node.
    - feature_importance (Optional[Dict[str, float]]): The importance of each feature, updated during tree construction.

    Methods:
    - is_leaf: Returns True if the current node is a leaf node (i.e., it does not split further).
    - predict: Predicts the output for a set of input samples.
    - to_dict: Converts the node into a dictionary representation for serialization.
    - from_dict: Restores the node from a dictionary representation (useful for deserialization).
    """

    NEGATIVE_INFINITY = float('-inf')

    def __init__(
        self,
        node_dict: Optional[Dict] = None,
        histogram: Optional[Dict[str, np.ndarray]] = None,
        feature_splits: Optional[Dict[str, np.ndarray]] = None,
        min_leaf: int = 5,
        min_child_weight: float = 1.0,
        depth: int = 10,
        lambda_: float = 1.0,
        gamma: float = 1.0,
        feature_importance: Optional[Dict[str, float]] = None,
    ):
        """
        Initializes a HistogramNode for use in a gradient boosting tree.

        Args:
        - node_dict (Optional[Dict], default: None): A dictionary to initialize the node (used for deserialization).
        - histogram (Optional[Dict], default: None): A dictionary containing histograms of gradients, hessians, and counts.
        - feature_splits (Optional[Dict], default: None): A dictionary containing possible splits for each feature.
        - min_leaf (int, default: 5): The minimum number of samples required in a leaf node.
        - min_child_weight (float, default: 1.0): The minimum sum of hessians for a child node.
        - depth (int, default: 10): The maximum depth of the tree.
        - lambda_ (float, default: 1.0): The L2 regularization parameter.
        - gamma (float, default: 1.0): The pruning parameter.
        - feature_importance (Optional[Dict[str, float]], default: None): A dictionary to store feature importance.
        """
        if node_dict is not None:
            self.from_dict(node_dict)
            return

        # Public attributes
        self.depth = depth
        self.feature_importance = feature_importance

        # Private attributes
        self._histogram = histogram or {}
        self._feature_splits = feature_splits or {}
        self._min_leaf = min_leaf
        self._lambda = lambda_
        self._gamma = gamma
        self._min_child_weight = min_child_weight
        self._score = self.NEGATIVE_INFINITY
        self._val = self._compute_gamma()

        # Split-related attributes
        self._var_idx = None
        self._split_idx = None
        self._split_value = None
        self._left_child = None
        self._right_child = None

        self._find_varsplit()

    def _compute_gamma(self) -> float:
        total_gradient = np.sum(self._histogram.get('gradients', []))
        total_hessian = np.sum(self._histogram.get('hessians', []))
        return -total_gradient / (total_hessian + self._lambda)

    def _find_varsplit(self):
        for feature_idx in range(len(self._feature_splits)):
            self._find_greedy_split(feature_idx)

        if self.is_leaf:
            return

        # Update feature importance if provided
        if self.feature_importance is not None and self._var_idx is not None:
            self.feature_importance[self._var_idx] = (
                self.feature_importance.get(self._var_idx, 0) + self._score
            )

        # Split feature ranges for child nodes
        lhs_splits = self._split_feature_ranges(self._feature_splits, self._var_idx, self._split_idx, left=True)
        rhs_splits = self._split_feature_ranges(self._feature_splits, self._var_idx, self._split_idx, left=False)

        # Set split value
        self._split_value = self._feature_splits[self._var_idx][self._split_idx]

        # Recursively create left and right child nodes
        self._left_child = HistogramNode(
            histogram=self._left_child_hist,
            feature_splits=lhs_splits,
            min_leaf=self._min_leaf,
            min_child_weight=self._min_child_weight,
            depth=self.depth - 1,
            lambda_=self._lambda,
            gamma=self._gamma,
            feature_importance=self.feature_importance,
        )
        self._right_child = HistogramNode(
            histogram=self._right_child_hist,
            feature_splits=rhs_splits,
            min_leaf=self._min_leaf,
            min_child_weight=self._min_child_weight,
            depth=self.depth - 1,
            lambda_=self._lambda,
            gamma=self._gamma,
            feature_importance=self.feature_importance,
        )

    def _find_greedy_split(self, feature_idx: int):
        gradients = self._histogram.get('gradients', [])
        hessians = self._histogram.get('hessians', [])
        counts = self._histogram.get('counts', [])

        num_splits = len(list(self._feature_splits.values())[feature_idx])
        for split_idx in range(num_splits):
            lhs_gradients, rhs_gradients = self._split_feature(gradients, feature_idx, split_idx + 1)
            lhs_hessians, rhs_hessians = self._split_feature(hessians, feature_idx, split_idx + 1)
            lhs_counts, rhs_counts = self._split_feature(counts, feature_idx, split_idx + 1)

            lhs_gradient, rhs_gradient = np.sum(lhs_gradients), np.sum(rhs_gradients)
            lhs_hessian, rhs_hessian = np.sum(lhs_hessians), np.sum(rhs_hessians)
            lhs_count, rhs_count = np.sum(lhs_counts), np.sum(rhs_counts)

            if (lhs_hessian < self._min_child_weight or
                rhs_hessian < self._min_child_weight or
                lhs_count < self._min_leaf or
                rhs_count < self._min_leaf):
                continue

            gain = self._compute_gain(lhs_gradient, lhs_hessian, rhs_gradient, rhs_hessian)
            if gain > self._score and gain > 0:
                self._var_idx = list(self._feature_splits.keys())[feature_idx]
                self._score = gain
                self._split_idx = split_idx
                self._left_child_hist = {
                    'gradients': lhs_gradients,
                    'hessians': lhs_hessians,
                    'counts': lhs_counts,
                }
                self._right_child_hist = {
                    'gradients': rhs_gradients,
                    'hessians': rhs_hessians,
                    'counts': rhs_counts,
                }

    def _compute_gain(self, lhs_gradient: float, lhs_hessian: float, rhs_gradient: float, rhs_hessian: float) -> float:
        gain = 0.5 * (
            (lhs_gradient ** 2 / (lhs_hessian + self._lambda)) +
            (rhs_gradient ** 2 / (rhs_hessian + self._lambda)) - 
            ((lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + self._lambda))
        ) - self._gamma
        return gain

    @staticmethod
    def _split_feature(array: np.ndarray, axis: int, index: int) -> Union[np.ndarray, np.ndarray]:
        split1 = array.take(indices=range(0, index), axis=axis)
        split2 = array.take(indices=range(index, array.shape[axis]), axis=axis)
        return split1, split2

    def _split_feature_ranges(self, feature_splits: Dict, var_idx: str, split_idx: int, left: bool) -> Dict:
        new_splits = feature_splits.copy()
        if left:
            new_splits[var_idx] = feature_splits[var_idx][:split_idx]
        else:
            new_splits[var_idx] = feature_splits[var_idx][split_idx + 1:]
        return new_splits

    @property
    def is_leaf(self) -> bool:
        return self._score == self.NEGATIVE_INFINITY or self.depth <= 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for a set of input samples.

        Args:
        - x (np.ndarray): The input samples to predict.

        Returns:
        - np.ndarray: The predicted values.
        """
        return np.array([self._predict_row(xi) for xi in x])

    def _predict_row(self, xi: np.ndarray) -> float:
        if self.is_leaf:
            return self._val

        next_node = self._left_child if xi[self._var_idx] <= self._split_value else self._right_child
        return next_node._predict_row(xi)

    def to_dict(self) -> Dict:
        """
        Converts the current node to a dictionary representation.

        Returns:
        - Dict: The dictionary representation of the node.
        """
        if self.is_leaf:
            return {'val': self._val, 'depth': self.depth, 'score': self._score}

        return {
            'var_idx': self._var_idx,
            'split_value': self._split_value,
            'score': self._score,
            'depth': self.depth,
            'left_child': self._left_child.to_dict(),
            'right_child': self._right_child.to_dict(),
        }

    def from_dict(self, node_dict: Dict):
        """
        Restores the node from a dictionary representation.

        Args:
        - node_dict (Dict): The dictionary containing the serialized node data.
        """
        if 'val' in node_dict:
            self._val = node_dict['val']
            self.depth = node_dict['depth']
            self._score = node_dict['score']
            return

        self._var_idx = node_dict['var_idx']
        self._split_value = node_dict['split_value']
        self._score = node_dict['score']
        self.depth = node_dict['depth']

        self._left_child = HistogramNode(node_dict=node_dict['left_child'])
        self._right_child = HistogramNode(node_dict=node_dict['right_child'])
