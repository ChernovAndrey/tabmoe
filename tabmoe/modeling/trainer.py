import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import math
import delu
from typing import Literal

from .model import Model
from .optimization import get_optimizer, make_parameter_groups

from tabmoe.utils.metrics import evaluate_predictions
from tabmoe.utils.model import get_n_parameters


class Trainer:
    def __init__(self, model: Model, optimizer_parameters: dict, train_batch_size: int,
                 eval_batch_size: None | int = 64000, gradient_clipping_norm: None | int = 1.0,
                 score_metric: None | Literal['accuracy', 'f1_macro', 'f1_micro', 'f1', 'rmse', 'r2'] = None):

        # TODO: maybe add torch.compile in future
        assert optimizer_parameters.get('type',
                                        None) is not None, "The optimizer_parameters dictionary must have a 'type' key"
        self.model = model
        self.train_batch_size = train_batch_size
        self.epoch_size = math.ceil(len(self.model.dataset.y_train) / self.train_batch_size)
        self.eval_batch_size = eval_batch_size

        self.optimizer = get_optimizer(**optimizer_parameters, params=make_parameter_groups(model))
        self.gradient_clipping_norm = gradient_clipping_norm
        self.loss_fn = (
            nn.functional.mse_loss
            if model.dataset.is_regression
            else nn.functional.binary_cross_entropy_with_logits
            if model.dataset.is_binary
            else nn.functional.cross_entropy
        )

        # TODO: make it more flexible
        if score_metric is None:
            self.score_metric = 'rmse' if self.model.dataset.is_regression else 'accuracy'
        else:
            self.score_metric = score_metric
            if self.model.dataset.is_regression:
                assert self.score_metric in ['rmse', 'r2'], \
                    "For regression tasks, only the following score metrics are supported: ['rmse', 'r2']"
            elif self.model.dataset.is_binary:
                assert self.score_metric in ['accuracy', 'f1'], \
                    "For binary classification tasks, only the following score metrics are supported:" \
                    " ['accuracy', 'f1']"
            else:
                assert self.score_metric in ['accuracy', 'f1_macro', 'f1_micro'], \
                    "For multiclass classification tasks, only the following score metrics are supported:" \
                    " ['accuracy', 'f1_macro', 'f1_micro']"

        # DataLoaders
        X_train = torch.cat(
            [X for X in (self.model.dataset.X_train_num, self.model.dataset.X_train_cat, self.model.dataset.X_train_bin)
             if X is not None], dim=1)

        self.train_loader = DataLoader(
            TensorDataset(X_train, self.model.dataset.y_train),
            batch_size=self.train_batch_size,
            shuffle=True if self.model.dataset.seed is None else torch.Generator().manual_seed(self.model.dataset.seed))

        if self.model.dataset.y_val is not None:
            X_val = torch.cat(
                [X for X in (self.model.dataset.X_val_num, self.model.dataset.X_val_cat, self.model.dataset.X_val_bin)
                 if X is not None], dim=1)
            self.val_loader = DataLoader(
                TensorDataset(X_val, self.model.dataset.y_val), batch_size=self.eval_batch_size, shuffle=False)

        if self.model.dataset.y_test is not None:
            X_test = torch.cat(
                [X for X in
                 (self.model.dataset.X_test_num, self.model.dataset.X_test_cat, self.model.dataset.X_test_bin)
                 if X is not None], dim=1)
            self.test_loader = DataLoader(
                TensorDataset(X_test, self.model.dataset.y_test), batch_size=self.eval_batch_size, shuffle=False)

        self.n_parameters = get_n_parameters(self.model)
        print(f'number parameters:{self.n_parameters}')
        # writer = torch.utils.tensorboard.SummaryWriter(output)  #TODO: add tensorboard? # type: ignore[code]

    # def calculate_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
    #     return self.loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(y_pred.shape[1]))

    def train(self, *, epochs: None | int = None, patience: None | int = None):
        assert (epochs is None) ^ (patience is None), \
            "Exactly one argument, either epochs or patience, must not be None, but not both"
        assert (patience is None or self.val_loader is not None), \
            "if patience is provided, a validation dataset must exist"
        timer = delu.tools.Timer()
        training_log = []

        early_stopping = delu.tools.EarlyStopping(patience, mode='max') if patience is not None else None

        epoch_i = 0
        val_score = -99999
        while True:
            self.model.train()
            total_loss = 0.0
            for X, y_batch in self.train_loader:
                output = self.model.run(X)
                loss = self.loss_fn(output, y_batch)
                # loss = self.calculate_loss(output, y_batch)
                loss.backward()

                # Apply gradient clipping if specified
                if self.gradient_clipping_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)

                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch_i + 1}, Train Loss: {total_loss / len(self.train_loader):.4f}")
            val_metrics = self.evaluate(self.val_loader)

            print('val_metrics:')
            print(val_metrics)
            if val_metrics['score'] > val_score:
                val_score = val_metrics['score']
                print(f'new best epoch: {epoch_i}, val_score: {val_score}')

            epoch_i += 1
            if early_stopping is not None:
                # TODO: get rid of hardcode
                early_stopping.update(-total_loss if self.score_metric in ['rmse'] else total_loss)

            if ((epochs is not None) and (epoch_i == epochs)) or \
                    ((early_stopping is not None) and (early_stopping.should_stop())):
                print(f'training is finished, {epoch_i} epochs were completed')
                break

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, num_samples: None | int = None, return_average: bool = True) -> dict:
        """
        Evaluates the model on a given DataLoader and computes performance metrics.

        Parameters:
        - loader: DataLoader, the data loader containing validation/test data.

        Returns:
        - dict: A dictionary with average loss and relevant evaluation metrics.
        """
        assert loader is not None

        self.model.eval()
        total_loss = 0.0
        all_y_true = []  # Store as a NumPy array
        all_y_pred = []

        for X, y_batch in loader:
            output = self.model.run(X, num_samples, return_average)
            print(output.shape)
            # Compute batch loss
            loss = self.loss_fn(output, y_batch)
            total_loss += loss.item()

            # Handle binary classification: Compare logit with zero
            if self.model.dataset.is_binary:
                preds = (output > 0).cpu().numpy().astype(int)
            # Handle multiclass classification: Get predicted class
            elif self.model.dataset.is_regression:
                preds = output.cpu().numpy()
            else:  # i.e. multiclass
                preds = torch.argmax(output, dim=-1).cpu().numpy()

            all_y_pred.append(preds)
            all_y_true.append(y_batch.cpu().numpy())

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)

        # Compute average loss per loader
        avg_loss = total_loss / len(loader)
        print(all_y_pred.shape)
        print(all_y_true.shape)
        metrics = evaluate_predictions(all_y_true, all_y_pred, self.model.dataset.task_type, self.score_metric)
        metrics["loss"] = avg_loss

        return metrics
