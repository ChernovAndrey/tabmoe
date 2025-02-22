import numpy as np
from typing import Self


class BinaryEncoder:
    """
    A transformer that converts binary categorical features into {0,1} encoding (as np.float32).
    Ensures each feature in X_train has exactly two unique values before encoding.
    """

    def __init__(self):
        self.mappings_ = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Fit the transformer by learning the mapping for each feature.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - self: fitted instance
        """
        n_features = X.shape[1]

        self.mappings_ = {}

        for i in range(n_features):
            unique_vals = np.unique(X[:, i])
            if len(unique_vals) != 2:
                raise ValueError(f"Binary Feature at column {i} does not have exactly two unique values.")
            self.mappings_[i] = {unique_vals[0]: 0, unique_vals[1]: 1}

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the dataset using the learned mappings.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - Transformed numpy array of the same shape, with values mapped to {0,1} as np.float32.
        """
        n_features = X.shape[1]

        if len(self.mappings_) != n_features:
            raise ValueError("Mismatch between fitted mappings and input features.")

        X_transformed = np.zeros_like(X, dtype=np.float32)

        for i in range(n_features):
            if not np.all(np.isin(X[:, i], list(self.mappings_[i].keys()))):
                raise ValueError(f"Binary feature at column {i} contains unseen values.")

            X_transformed[:, i] = np.vectorize(self.mappings_[i].get)(X[:, i]).astype(np.float32)

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform the dataset in one step.
        """
        return self.fit(X).transform(X)


