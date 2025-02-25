from enum import Enum

class EmbeddingPolicy(Enum):
    PIECEWISE_LINEAR_EMBEDDINGS = "piecewise_linear_embeddings"

class GatingType(Enum):
    NONE = "standard"
    PIECEWISE_LINEAR_EMBEDDINGS = "gumbel"
