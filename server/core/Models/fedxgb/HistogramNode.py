import numpy as np
from typing import Dict, List, Optional, Union

class HistogramNode:
    """
    A node object for constructing a regression tree using histograms of gradients and hessians.
    
    Attributes:
        depth (int): The maximum depth of the tree.
        feature_importance (Optional[Dict[str, float]]): A dictionary to store feature importance scores.
    
    Methods:
        predict(x: np.ndarray) -> np.ndarray: Predicts the target values for input data.
        to_dict() -> Dict: Converts the node and its children into a dictionary representation.
        from_dict(node_dict: Dict): Reconstructs a node from a dictionary representation.
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
        Initializes a HistogramNode.

        Args:
            node_dict (Optional[Dict]): A dictionary representation of a node (used for reconstruction).
            histogram (Optional[Dict[str, np.ndarray]]): Histogram data containing gradients, hessians, and counts.
            feature_splits (Optional[Dict[str, np.ndarray]]): Possible split points for each feature.
            min_leaf (int): Minimum number of samples required for a node to be valid.
            min_child_weight (float): Minimum sum of hessians required to allow a split.
            depth (int): Maximum depth of the tree.
            lambda_ (float): L2 regularization term on weights.
            gamma (float): Minimum gain required to make a split.
            feature_importance (Optional[Dict[str, float]]): Dictionary for tracking feature importance.

        Raises:
            ValueError: If the histogram is improperly formatted or missing required keys.
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
        """
        Calculates the optimal leaf value using the sum of gradients and hessians.

        Returns:
            float: The optimal leaf value.
        """
        total_gradient = np.sum(self._histogram.get('gradients', []))
        total_hessian = np.sum(self._histogram.get('hessians', []))
        return -total_gradient / (total_hessian + self._lambda)

    def _find_varsplit(self):
        """
        Identifies the best feature and split point for the current node.
        Updates child nodes if a valid split is found.
        """
        for feature_idx in range(len(self._feature_splits)):
            self._find_greedy_split(feature_idx)

        if self.is_leaf:
            return

        # Update feature importance
        if self.feature_importance is not None and self._var_idx is not None:
            self.feature_importance[self._var_idx] = (
                self.feature_importance.get(self._var_idx, 0) + self._score
            )

        # Split feature ranges
        lhs_splits = self._split_feature_ranges(self._feature_splits, self._var_idx, self._split_idx, left=True)
        rhs_splits = self._split_feature_ranges(self._feature_splits, self._var_idx, self._split_idx, left=False)

        # Set split value
        self._split_value = self._feature_splits[self._var_idx][self._split_idx]

        # Recursive child creation
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
        """
        Evaluates potential splits for a given feature and selects the one with the highest gain.

        Args:
            feature_idx (int): Index of the feature to evaluate.
        """
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
        """
        Computes the gain for a potential split based on the XGBoost formula.

        Args:
            lhs_gradient (float): Sum of gradients for the left child.
            lhs_hessian (float): Sum of hessians for the left child.
            rhs_gradient (float): Sum of gradients for the right child.
            rhs_hessian (float): Sum of hessians for the right child.

        Returns:
            float: The computed gain for the split.
        """
        gain = 0.5 * (
            (lhs_gradient ** 2 / (lhs_hessian + self._lambda)) +
            (rhs_gradient ** 2 / (rhs_hessian + self._lambda)) -
            ((lhs_gradient + rhs_gradient) ** 2 / (lhs_hessian + rhs_hessian + self._lambda))
        ) - self._gamma
        return gain

    @staticmethod
    def _split_feature(array: np.ndarray, axis: int, index: int) -> Union[np.ndarray, np.ndarray]:
        """
        Splits an array along a specified axis at the given index.

        Args:
            array (np.ndarray): The array to split.
            axis (int): The axis along which to split.
            index (int): The split index.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The left and right splits of the array.
        """
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
        """
        Determines if the current node is a leaf.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self._score == self.NEGATIVE_INFINITY or self.depth <= 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for input data.

        Args:
            x (np.ndarray): Input data array.

        Returns:
            np.ndarray: Predicted target values.
        """
        return np.array([self._predict_row(xi) for xi in x])

    def _predict_row(self, xi: np.ndarray) -> float:
        """
        Predicts the target value for a single data point.

        Args:
            xi (np.ndarray): A single data point.

        Returns:
            float: Predicted target value.
        """
        if self.is_leaf:
            return self._val

        next_node = self._left_child if xi[self._var_idx] <= self._split_value else self._right_child
        return next_node._predict_row(xi)

    def to_dict(self) -> Dict:
        """
        Converts the node and its children into a dictionary representation.

        Returns:
            Dict: A dictionary representation of the node.
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
