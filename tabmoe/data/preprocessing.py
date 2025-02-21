import enum
import hashlib
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, FunctionTransformer
# from loguru import logger
from tabmoe.enums.utils import validate_enum
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Optional
from tabmoe.enums.data_processing import TaskType, FeatureType, NumPolicy, EmbeddingPolicy
import rtdl_num_embeddings
import torch


class Dataset:
    def __init__(self,
                 X_train: np.array, y_train: np.array,
                 X_types: list[str], task_type: str,
                 num_policy: str = "NOISY_QUANTILE",
                 embedding_policy: str = "None",
                 X_val: Optional[np.array] = None, y_val: Optional[np.array] = None,
                 X_test: Optional[np.array] = None, y_test: Optional[np.array] = None,
                 n_bins: Optional[int] = None,
                 seed: Optional[int] = None, ):
        if not isinstance(X_types, list):
            raise TypeError(f"Expected a list, but got {type(X_types).__name__}")

        assert X_train.shape[1] == len(X_types), "X type must be provided for every feature in X_train"
        assert len(y_train.shape) == 1, "We support only 1D labels"

        self.task_type: TaskType = validate_enum(TaskType, task_type)
        self.x_types: list[FeatureType] = [validate_enum(FeatureType, value) for value in X_types]
        self.num_policy: NumPolicy = validate_enum(NumPolicy, num_policy)
        self.embedding_policy: EmbeddingPolicy = validate_enum(EmbeddingPolicy, embedding_policy)
        self.n_bins = n_bins
        self.seed = seed

        # Store datasets
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        # Initialize label scaler for regression tasks
        self.label_scaler = StandardScaler() if self.task_type == TaskType.REGRESSION else None

        # Identify feature types
        self.categorical_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.CATEGORICAL]
        self.binary_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.BINARY]
        self.numeric_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.NUMERIC]
        print(f"numeric indices:{self.numeric_indices}")
        # Compute bins for Piecewise Linear Embeddings if needed
        self.bin_edges = None
        if self.embedding_policy == EmbeddingPolicy.PIECEWISE_LINEAR_EMBEDDINGS:
            if self.n_bins is None:
                raise ValueError("Number of bins (`n_bins`) must be specified when using PiecewiseLinearEmbeddings.")

            # TODO: convert to tensors everything before embeddings?
            self.bin_edges = rtdl_num_embeddings.compute_bins(
                torch.tensor(self.X_train[:, self.numeric_indices], dtype=torch.float32), self.n_bins)

        # Define preprocessing pipeline
        self.preprocessor = self._build_preprocessing_pipeline()

    def _build_preprocessing_pipeline(self):
        """Builds a Scikit-Learn ColumnTransformer pipeline for preprocessing."""

        # One-Hot Encoding for categorical features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                                dtype=np.float32)  # TODO: why is sparse not expected?

        # Mapping binary features to {0,1}
        binary_transformer = FunctionTransformer(self._map_binary_features)

        # Numerical transformation: StandardScaler or QuantileTransformer
        if self.num_policy == NumPolicy.STANDARD:
            numeric_transformer = StandardScaler()
        elif self.num_policy == NumPolicy.NOISY_QUANTILE:
            numeric_transformer = QuantileTransformer(
                n_quantiles=max(min(len(self.X_train) // 30, 1000), 10),
                output_distribution='normal',
                subsample=1_000_000_000,
                random_state=self.seed
            )

        # ColumnTransformer to apply transformations to respective columns
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, self.numeric_indices),
            ("cat", categorical_transformer, self.categorical_indices),
            ("bin", binary_transformer, self.binary_indices),
        ])

        return Pipeline(steps=[("preprocessor", preprocessor)])

    def _map_binary_features(self, X):
        """Ensures binary features are mapped to {0,1}."""
        for col in range(X.shape[1]):
            unique_values = np.unique(X[:, col])
            if len(unique_values) != 2:
                raise ValueError(f"Binary feature at index {col} has unexpected values: {unique_values}")
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            X[:, col] = np.vectorize(mapping.get)(X[:, col])
        return X

    def preprocess(self) -> None:
        """Preprocess training, validation, and test datasets using the pipeline."""
        # if self.num_policy == NumPolicy.NOISY_QUANTILE:  # TODO: improve clarity?, get rid of copying!
        #     X_train_num = self.X_train[:, self.numeric_indices]
        #     noise = np.random.normal(0.0, 1e-5, X_train_num.shape).astype(
        #         X_train_num.dtype) if self.seed is None else \
        #         np.random.RandomState(self.seed).normal(0.0, 1e-5, X_train_num.shape).astype(X_train_num.dtype)
        # else:
        #     noise = 0.0
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        if self.X_val is not None:
            self.X_val = self.preprocessor.transform(self.X_val)
        if self.X_test is not None:
            self.X_test = self.preprocessor.transform(self.X_test)

        # Standardize labels for regression
        if self.task_type == TaskType.REGRESSION:
            self.y_train = self.label_scaler.fit_transform(self.y_train.reshape(-1, 1)).reshape(-1)
            if self.y_val is not None:
                self.y_val = self.label_scaler.transform(self.y_val.reshape(-1, 1)).reshape(-1)
            if self.y_test is not None:
                self.y_test = self.label_scaler.transform(self.y_test.reshape(-1, 1)).reshape(-1)

    def transform(self, X_new: np.array) -> np.array:
        """
        Transforms new test data using the already fitted preprocessing pipeline.

        Args:
            X_new (np.array): New input data.

        Returns:
            np.array: Preprocessed version of X_new.
        """
        if X_new.shape[1] != len(self.x_types):
            raise ValueError("New data must have the same number of features as the training data.")
        return self.preprocessor.transform(X_new)
