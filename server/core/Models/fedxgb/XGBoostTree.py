import numpy as np
from .HistogramNode import HistogramNode
from typing import Optional, Dict, Any

class XGBoostTree:
    """
    A class representing a gradient boosting tree. This class provides an interface to create and use 
    a regression tree for gradient boosting using histograms for efficient computation.

    Attributes
    ----------
    dtree : Optional[HistogramNode]
        The root node of the decision tree.

    Methods
    -------
    fit(x: pd.DataFrame, gradient: np.ndarray, hessian: np.ndarray, 
        subsample_cols: float = 0.8, min_leaf: int = 5, min_child_weight: int = 1, 
        depth: int = 10, lambda_: float = 1, gamma: float = 1) -> 'XGBoostTree':
        Trains the decision tree using the provided training data.

    hist_fit(node_dict: Optional[Dict[str, Any]] = None, histogram: Optional[np.ndarray] = None, 
             feature_splits: Optional[np.ndarray] = None, min_leaf: int = 5, min_child_weight: int = 1, 
             depth: int = 10, lambda_: float = 1, gamma: float = 1, feature_importance: Optional[np.ndarray] = None) -> 'XGBoostTree':
        Fits the tree using histogram-based methods (useful for approximate tree construction).

    predict(X: np.ndarray) -> np.ndarray:
        Makes predictions on the input data X.

    to_dict() -> Dict[str, Any]:
        Converts the tree to a dictionary representation.

    from_dict(tree_dict: Dict[str, Any]) -> 'XGBoostTree':
        Initializes the tree from a dictionary representation.
    """
    
    def __init__(self):
        """
        Initializes an empty tree structure with no root node.
        """
        self.dtree: Optional[HistogramNode] = None

    def hist_fit(self, node_dict: Optional[Dict[str, Any]] = None, histogram: Optional[np.ndarray] = None, 
                 feature_splits: Optional[np.ndarray] = None, min_leaf: int = 5, 
                 min_child_weight: int = 1, depth: int = 10, lambda_: float = 1, 
                 gamma: float = 1, feature_importance: Optional[np.ndarray] = None) -> 'XGBoostTree':
        """
        Fit the tree using histograms or a provided dictionary representation.
        
        Parameters
        ----------
        node_dict : dict, optional
            A dictionary representation of the tree.
        histogram : np.ndarray, optional
            Histogram data for constructing the tree.
        feature_splits : np.ndarray, optional
            The split points for features.
        min_leaf : int, default=5
            Minimum number of samples per leaf.
        min_child_weight : int, default=1
            Minimum sum of Hessian per child node.
        depth : int, default=10
            Maximum tree depth.
        lambda_ : float, default=1
            L2 regularization term on weights.
        gamma : float, default=1
            Minimum gain required to make a split.
        feature_importance : np.ndarray, optional
            Feature importance values.

        Returns
        -------
        XGBoostTree
            The fitted tree object.
        """
        if node_dict is not None:
            self.dtree = self.from_dict(node_dict)
        else:
            self.dtree = HistogramNode(histogram=histogram, feature_splits=feature_splits,
                                       min_leaf=min_leaf, min_child_weight=min_child_weight,
                                       depth=depth, lambda_=lambda_, gamma=gamma, 
                                       feature_importance=feature_importance)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained tree model.

        Parameters
        ----------
        X : np.ndarray
            The input data for prediction.

        Returns
        -------
        np.ndarray
            The predicted values for the input data.
        """
        if self.dtree is None:
            raise ValueError("The tree is not trained yet. Please call fit() first.")
        return self.dtree.predict(X)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the tree to a dictionary representation.

        Returns
        -------
        dict
            The dictionary representation of the tree.
        """
        if self.dtree is None:
            raise ValueError("The tree is not trained yet. Please call fit() first.")
        return self.dtree.to_dict()

    def from_dict(self, tree_dict: Dict[str, Any]) -> 'XGBoostTree':
        """
        Initializes the tree from a dictionary representation.

        Parameters
        ----------
        tree_dict : dict
            The dictionary representation of the tree.

        Returns
        -------
        XGBoostTree
            The tree initialized from the dictionary.
        """
        self.dtree = HistogramNode(node_dict=tree_dict)
        return self
