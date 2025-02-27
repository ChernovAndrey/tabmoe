from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

from tabmoe.enums.data_processing import TaskType
from typing import Literal


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, task_type: TaskType,
                         score: Literal['accuracy', 'f1_macro', 'f1_micro', 'f1', 'rmse', 'r2']):
    """
    Evaluate classification (binary/multiclass) and regression performance.

    Parameters:
    - y_true: array-like, Ground truth labels or values.
    - y_pred: array-like, Predicted labels (for classification) or values (for regression).

    Returns:
    - dict: A dictionary containing relevant performance metrics.
    """
    metrics = {}

    # Binary or Multiclass Classification
    if task_type == TaskType.MULTICLASS:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(y_true, y_pred, average='macro')
        metrics["recall_macro"] = recall_score(y_true, y_pred, average='macro')
        metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro')
        metrics["precision_micro"] = precision_score(y_true, y_pred, average='micro')
        metrics["recall_micro"] = recall_score(y_true, y_pred, average='micro')
        metrics["f1_micro"] = f1_score(y_true, y_pred, average='micro')

    elif task_type == TaskType.BINCLASS:  # Binary Classification
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred)

    else:  # Regression
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["r2"] = r2_score(y_true, y_pred)

    metrics['score'] = metrics[score]
    return metrics
