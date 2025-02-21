from enum import Enum


class TaskType(Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


class FeatureType(Enum):
    CATEGORICAL = 'categorical'
    BINARY = 'binary'
    NUMERIC = 'numeric'


class NumPolicy(Enum):
    # preprocessing policies for numerical features
    STANDARD = "standard"
    NOISY_QUANTILE = "noisy_quantile"


class EmbeddingPolicy(Enum):
    NONE = "none"
    PIECEWISE_LINEAR_EMBEDDINGS = "piecewise_linear_embeddings"
