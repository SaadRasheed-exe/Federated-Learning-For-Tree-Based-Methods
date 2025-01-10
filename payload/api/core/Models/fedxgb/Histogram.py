import numpy as np
from typing import Dict, List, Tuple, Any
 
 
class Histogram:
    """
    A class to compute histograms over data regions defined by feature splits.
 
    Attributes:
        feature_splits (dict): A dictionary mapping feature indices to their respective split points.
        splits_per_feature (dict): A dictionary mapping feature indices to the number of splits.
        feature_masks (dict): A dictionary mapping features to their precomputed masks.
        regions (dict): A dictionary mapping region tuples to lists of data indices.
        histogram (dict): A dictionary containing computed gradients, hessians, and counts.
    """
 
    def __init__(self, feature_splits: Dict[int, List]) -> None:
        """
        Initializes the Histogram class.
 
        Args:
            feature_splits (dict): A dictionary where keys are feature indices
                                   and values are numpy arrays of split points.
        """
        if not isinstance(feature_splits, dict) or not all(isinstance(splits, List) for splits in feature_splits.values()):
            raise ValueError("feature_splits must be a dictionary of feature indices to numpy arrays of split points.")
 
        self.feature_splits = feature_splits
        self.splits_per_feature = {feature: len(splits) for feature, splits in feature_splits.items()}
        self.feature_masks = None
        self._regions_cache = {}
 
    def fit(self, X: np.ndarray) -> None:
        """
        Fits the data by computing the feature masks and storing the number of rows.
 
        Args:
            X (np.ndarray): The input dataset with features as columns.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
 
        self.rows = X.shape[0]
        self.feature_masks = self._compute_feature_masks(X)
 
    def _compute_feature_masks(self, data: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """
        Precomputes masks for each feature's split.
 
        Args:
            data (np.ndarray): The dataset.
 
        Returns:
            dict: A dictionary mapping feature indices to masks.
        """
        feature_masks = {}
        for feature, splits in self.feature_splits.items():
            if not splits:
                continue
 
            masks = []
            for i, split in enumerate(splits):
                lower_mask = (data[:, feature] > splits[i - 1]) if i > 0 else np.ones(data.shape[0], dtype=bool)
                upper_mask = (data[:, feature] <= split)
                masks.append(lower_mask & upper_mask)
            masks.append(data[:, feature] > splits[-1])
            feature_masks[feature] = masks
        return feature_masks
 
    def _get_region_indices(self, features: List[int]) -> Dict[Tuple[Tuple[int, int]], List[int]]:
        """
        Identifies the indices of dataset points in each region defined by feature splits.
 
        Args:
            features (list): List of feature indices to compute regions for.
 
        Returns:
            dict: A dictionary mapping region tuples to lists of indices.
        """
        regions = {}
 
        def combine_masks(feature_idx: int, current_mask: np.ndarray, region_key: List[Tuple[int, int]]) -> None:
            """
            Recursively combines feature masks to assign indices to regions.
            """
            if feature_idx == len(features):
                regions[tuple(region_key)] = np.where(current_mask)[0].tolist()
                return
 
            feature = features[feature_idx]
            for i, mask in enumerate(self.feature_masks[feature]):
                new_mask = current_mask & mask
                new_region_key = region_key + [(feature, i)]
                combine_masks(feature_idx + 1, new_mask, new_region_key)
 
        combine_masks(0, np.ones(self.rows, dtype=bool), [])
        return regions
 
    def compute_histogram(self, Grad: np.ndarray, Hess: np.ndarray, features_subset: List[int], compute_regions: bool = True) -> Dict[str, Any]:
        """
        Computes the histogram of gradients and hessians for each region.
 
        Args:
            Grad (np.ndarray): Array of gradients.
            Hess (np.ndarray): Array of hessians.
            features_subset (list): List of feature indices to consider for histogram computation.
            compute_regions (bool): Whether to compute regions before histogram computation.
 
        Returns:
            dict: A dictionary containing gradients, hessians, and counts for each region.
        """
        if self.feature_splits is None:
            raise RuntimeError("Feature splits are not defined. Call `fit` first to initialize the feature splits.")
       
        id = tuple(features_subset)
 
        if compute_regions and id not in self._regions_cache:
            regions = self._get_region_indices(features_subset)
            self._regions_cache[id] = regions
        elif id in self._regions_cache:
            regions = self._regions_cache[id]
        else:
            raise ValueError("Regions cache is empty. Set `compute_regions` to True to compute regions")
 
        splits_per_feature = {feature: self.splits_per_feature[feature] for feature in features_subset}
 
        total_regions = len(regions)
        gradients = np.zeros(total_regions)
        hessians = np.zeros(total_regions)
        counts = np.zeros(total_regions)
 
        for i, (_, indices) in enumerate(regions.items()):
            gradients[i] = np.sum(Grad[indices])
            hessians[i] = np.sum(Hess[indices])
            counts[i] = len(indices)
 
        shape = np.array(list(splits_per_feature.values())) + 1
        return {
            "gradients": gradients.reshape(shape),
            "hessians": hessians.reshape(shape),
            "counts": counts.reshape(shape),
        }