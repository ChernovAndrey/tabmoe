from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    Ensures that every model follows a common structure.
    """

    def __init__(self, **kwargs):
        """
        Base constructor for models. Accepts arbitrary parameters.
        """
        self.params = kwargs  # Store parameters for flexibility

    @abstractmethod
    def train(self, X, y, **kwargs):
        """
        Train the model. The implementation varies per model.

        Args:
            X: Feature matrix
            y: Target variable
            kwargs: Additional parameters (e.g., learning rate)
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs) -> Any:
        """
        Make predictions.

        Args:
            X: Feature matrix
            kwargs: Additional parameters (e.g., return confidence intervals)

        Returns:
            Predicted values
        """
        pass

    def save(self, path: str):
        """
        Save model parameters to a file.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        """
        Load model parameters from a file.
        """
        import pickle
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))