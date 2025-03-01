import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
import numpy as np
import math
import delu
import time
from typing import Literal
from enum import Enum

from .model import Model
from .optimization import get_optimizer, make_parameter_groups

from tabmoe.utils.metrics import evaluate_predictions
from tabmoe.utils.model import get_n_parameters
from tabmoe.utils.hyperparam_logger import HyperparamLogger
from tabmoe.utils.model import get_gpu_names


class Trainer:
    def __init__(self, model: Model, optimizer_parameters: dict, train_batch_size: int,
                 eval_batch_size: None | int = 64000, gradient_clipping_norm: None | int = 1.0,
                 score_metric: None | Literal['accuracy', 'f1_macro', 'f1_micro', 'f1', 'rmse', 'r2'] = None,
                 save_dir: str | Path = "", param_logger: HyperparamLogger = None):

        # TODO: maybe add torch.compile in future
        assert optimizer_parameters.get('type',
                                        None) is not None, "The optimizer_parameters dictionary must have a 'type' key"
        self.param_logger = param_logger
        if self.param_logger is not None:
            self.param_logger.log("trainer", optimizer_parameters=optimizer_parameters,
                                  train_batch_size=train_batch_size, eval_batch_size=eval_batch_size,
                                  gradient_clipping_norm=gradient_clipping_norm, score_metric=score_metric,
                                  save_dir=str(save_dir))
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
        print(f'number parameters: {self.n_parameters}')
        # writer = torch.utils.tensorboard.SummaryWriter(output)  #TODO: add tensorboard?

        # saving
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_path = self.save_dir / 'best_checkpoint.pth'
        self.save_report_path = self.save_dir / 'report.json'



    def train(self, *, epochs: None | int = None, patience: None | int = None):
        assert (epochs is None) ^ (patience is None), \
            "Exactly one argument, either epochs or patience, must not be None, but not both"
        assert (patience is None or self.val_loader is not None), \
            "if patience is provided, a validation dataset must exist"
        if self.param_logger is not None:
            self.param_logger.log('trainer', patience=patience, epochs=epochs)

        early_stopping = delu.tools.EarlyStopping(patience, mode='max') if patience is not None else None

        epoch_i = 0
        val_score = None
        best_epoch = None
        start_time = time.perf_counter()
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

            epoch_i += 1

            print(f"Epoch {epoch_i}, Train Loss: {total_loss / len(self.train_loader):.4f}")
            val_metrics = self.evaluate(self.val_loader)

            print(f"current val score: {val_metrics['score']}")
            if (val_score is None) or (val_metrics['score'] > val_score):  # TODO: WTF
                val_score = val_metrics['score']
                best_epoch = epoch_i
                print(f'----------------- hip, hip, hurrah! New best epoch: {epoch_i}, val_score: {val_score}')
                torch.save(self.model.state_dict(), self.save_model_path)


            if early_stopping is not None:
                # TODO: get rid of hardcode metrics
                early_stopping.update(val_score)

            if ((epochs is not None) and (epoch_i == epochs)) or \
                    ((early_stopping is not None) and (early_stopping.should_stop())):
                print(f'training is finished, {epoch_i} epochs were completed')
                print(f'the best val score: {val_score}')

                break

        training_time = time.perf_counter() - start_time
        self.model.load_state_dict(torch.load(self.save_model_path))  # loading the best checkpoint
        start_time = time.perf_counter()
        train_metrics = self.evaluate(self.train_loader)
        val_metrics = self.evaluate(self.val_loader) if self.val_loader is not None else None
        test_metrics = self.evaluate(self.test_loader) if self.test_loader is not None else None
        evaluation_time = time.perf_counter() - start_time

        print('train metrics:')
        print(train_metrics)

        print('val metrics:')
        print(val_metrics)

        print('test metrics:')
        print(test_metrics)
        if self.param_logger is not None:
            self.param_logger.log("metrics", training_time=training_time, evaluation_time=evaluation_time,
                                  train_metrics=train_metrics, val_metrics=val_metrics, test_metrics=test_metrics,
                                  best_epoch=best_epoch)

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
        if self.model.dataset.label_scaler is not None:
            all_y_true = self.model.dataset.label_scaler.inverse_transform(all_y_true.reshape(-1, 1)).reshape(-1)
            all_y_pred = self.model.dataset.label_scaler.inverse_transform(all_y_pred.reshape(-1, 1)).reshape(-1)

        metrics = evaluate_predictions(all_y_true, all_y_pred, self.model.dataset.task_type, self.score_metric)
        metrics["loss"] = avg_loss

        return metrics
