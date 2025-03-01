import torch
import numpy as np
from typing import Literal, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer

from tabmoe.enums.utils import validate_enum
from tabmoe.enums.data_processing import TaskType, FeatureType, NumPolicy
from tabmoe.preprocessing.binary_encoder import BinaryEncoder
from tabmoe.utils.model import get_device
from tabmoe.utils.hyperparam_logger import HyperparamLogger


class Dataset:
    def __init__(self,
                 X_train: np.array, y_train: np.array,
                 X_types: list[str], task_type: Literal['regression', 'multiclass', 'binclass'],
                 num_policy: Literal["noisy_quantile", "standardization"] = 'noisy_quantile',
                 X_val: Optional[np.array] = None, y_val: Optional[np.array] = None,
                 X_test: Optional[np.array] = None, y_test: Optional[np.array] = None,
                 seed: Optional[int] = None,
                 device: None | str | torch.device = None,
                 input_logger: None | HyperparamLogger = None):
        if not isinstance(X_types, list):
            raise TypeError(f"Expected a list, but got {type(X_types).__name__}")

        assert X_train.shape[1] == len(X_types), "X type must be provided for every feature in X_train"
        assert len(y_train.shape) == 1, "Only 1D labels are supported"

        if input_logger is not None:
            input_logger.log('dataset', X_types=X_types, num_policy=num_policy, seed=seed, device=device)

        self.task_type: TaskType = validate_enum(TaskType, task_type)
        self.x_types: list[FeatureType] = [validate_enum(FeatureType, value) for value in X_types]
        self.num_policy: NumPolicy = validate_enum(NumPolicy, num_policy)
        self.device = get_device() if device is None else device
        print(f'device:{self.device}')
        self.seed = seed

        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        # Initialize label scaler for regression tasks
        self.label_scaler = StandardScaler() if self.is_regression else None

        # Identify feature types
        self.cat_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.CATEGORICAL]
        self.bin_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.BINARY]
        self.num_indices = [i for i, f in enumerate(self.x_types) if f == FeatureType.NUMERIC]

        # Split features explicitly before preprocessing
        self.X_train_num, self.X_train_cat, self.X_train_bin = self._split_data(X_train)
        if X_val is not None:
            self.X_val_num, self.X_val_cat, self.X_val_bin = self._split_data(X_val)
        else:
            self.X_val_num, self.X_val_cat, self.X_val_bin = None, None, None
        if X_test is not None:
            self.X_test_num, self.X_test_cat, self.X_test_bin = self._split_data(X_test)
        else:
            self.X_test_num, self.X_test_cat, self.X_test_bin = None, None, None

        # One-Hot Encoding for categorical features
        self.cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                             dtype=np.float32)
        # Mapping binary features to {0,1}
        self.bin_transformer = BinaryEncoder()

        # Numeric Transformer for Categorical features
        if self.num_policy == NumPolicy.STANDARD:
            self.num_transformer = StandardScaler()
        elif self.num_policy == NumPolicy.NOISY_QUANTILE:
            self.num_transformer = QuantileTransformer(
                n_quantiles=max(min(self.X_train_num.shape[0] // 30, 1000), 10),
                output_distribution='normal',
                subsample=1_000_000_000,
                random_state=self.seed
            )

    def _split_data(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Splits X into numerical, categorical, and binary features."""
        X_num = np.array(X[:, self.num_indices], dtype=np.float32) if self.num_indices else None
        X_cat = X[:, self.cat_indices] if self.cat_indices else None
        X_bin = X[:, self.bin_indices] if self.bin_indices else None
        return X_num, X_cat, X_bin

    def preprocess(self) -> None:

        # preprocess numeric features
        if self.num_policy == NumPolicy.NOISY_QUANTILE:
            self.X_train_num += np.random.normal(0.0, 1e-5, self.X_train_num.shape).astype(
                self.X_train_num.dtype) if self.seed is None else \
                np.random.RandomState(self.seed).normal(0.0, 1e-5, self.X_train_num.shape).astype(
                    self.X_train_num.dtype)

        self.X_train_num = self.num_transformer.fit_transform(
            self.X_train_num) if self.X_train_num is not None else None

        self.X_val_num = self.num_transformer.transform(self.X_val_num) if self.X_val_num is not None else None
        self.X_test_num = self.num_transformer.transform(self.X_test_num) if self.X_test_num is not None else None

        # preprocess categorical features
        self.X_train_cat = self.cat_transformer.fit_transform(
            self.X_train_cat) if self.X_train_cat is not None else None

        self.X_val_cat = self.cat_transformer.transform(self.X_val_cat) if self.X_val_cat is not None else None
        self.X_test_cat = self.cat_transformer.transform(self.X_test_cat) if self.X_test_cat is not None else None

        # preprocess binary features
        # preprocess categorical features
        self.X_train_bin = self.bin_transformer.fit_transform(
            self.X_train_bin) if self.X_train_bin is not None else None

        self.X_val_bin = self.bin_transformer.transform(self.X_val_bin) if self.X_val_bin is not None else None
        self.X_test_bin = self.bin_transformer.transform(self.X_test_bin) if self.X_test_bin is not None else None

        # Standardize labels for regression
        if self.is_regression:
            self.y_train = self.label_scaler.fit_transform(self.y_train.reshape(-1, 1)).reshape(-1)
            if self.y_val is not None:
                self.y_val = self.label_scaler.transform(self.y_val.reshape(-1, 1)).reshape(-1)
            if self.y_test is not None:
                self.y_test = self.label_scaler.transform(self.y_test.reshape(-1, 1)).reshape(-1)

        self._to_torch(self.device)
        print('preprocessing is finished; data was converted to torch.tensor')

    def transform(self, X: np.ndarray, device: None | str | torch.device = None) -> Tuple[
        torch.tensor, torch.tensor, torch.tensor]:
        """
        Transforms new test preprocessing using the already fitted preprocessing pipeline.

        Args:
            X (np.ndarray): New input preprocessing.

        Returns:
            np.ndarray: Preprocessed version of X_new.
        """
        if X.shape[1] != len(self.x_types):
            raise ValueError("New preprocessing must have the same number of features as the training preprocessing.")
        X_num, X_cat, X_bin = self._split_data(X)

        X_num = self.num_transformer.transform(X_num) if X_num is not None else None
        X_cat = self.cat_transformer.transform(X_cat) if X_cat is not None else None
        X_bin = self.bin_transformer.transform(X_bin) if X_bin is not None else None

        if device is None:
            device = self.device
        return torch.tensor(X_num, dtype=torch.float32, device=device), \
            torch.tensor(X_cat, dtype=torch.float32, device=device), \
            torch.tensor(X_bin, dtype=torch.float32, device=device)

    def _to_torch(self, device: None | str | torch.device) -> None:
        for attr in ["X_train_num", "X_train_cat", "X_train_bin",
                     "X_val_num", "X_val_cat", "X_val_bin",
                     "X_test_num", "X_test_cat", "X_test_bin"]:
            setattr(self, attr,
                    torch.tensor(getattr(self, attr), dtype=torch.float32, device=device)
                    if getattr(self, attr) is not None else None)

        for attr in ["y_train", "y_val", "y_test"]:
            if getattr(self, attr) is not None:
                dtype = torch.float32 if self.is_regression or self.is_binary else torch.long
                setattr(self, attr, torch.tensor(getattr(self, attr), dtype=dtype, device=device))

    @property
    def n_num_features(self) -> int:
        return len(self.num_indices)

    @property
    def n_bin_features(self) -> int:
        return len(self.bin_indices)

    @property
    def n_cat_features(self) -> int:
        return len(self.cat_indices)

    @property
    def n_encoded_cat_features(self) -> int:
        if self.X_train_cat is None:
            return 0
        else:
            return self.X_train_cat.shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    @property
    def n_classes(self) -> None | int:
        return None if self.task_type == TaskType.REGRESSION else len(np.unique(self.y_train))

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def is_binary(self) -> bool:
        return self.task_type == TaskType.BINCLASS
