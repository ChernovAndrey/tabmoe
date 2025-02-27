import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

import math
import delu

from .model import Model
from .optimization import get_optimizer, make_parameter_groups

class Trainer:
    def __init__(self, model: Model, optimizer_parameters: dict, train_batch_size: int,
                eval_batch_size: None | int = 64000, gradient_clipping_norm: None | int = 1.0,
                patience: int = 16):
        # TODO: maybe add torch.compile in future
        assert optimizer_parameters.get('type',
                                        None) is not None, "The optimizer_parameters dictionary must have a 'type' key"
        self.step = 0
        self.model = model
        self.train_batch_size = train_batch_size
        self.epoch_size = math.ceil(len(self.model.dataset.y_train) / self.train_batch_size)
        self.eval_batch_size = eval_batch_size
        self.chunk_size = None  # TODO:?

        self.optimizer = get_optimizer(**optimizer_parameters, params=make_parameter_groups(model))
        self.gradient_clipping_norm = gradient_clipping_norm
        self.loss_fn = (
            nn.functional.mse_loss
            if model.dataset.is_regression
            else nn.functional.cross_entropy
        )

        # DataLoaders
        self.train_loader = DataLoader(
            TensorDataset(self.model.dataset.X_train_num, self.model.dataset.X_train_cat,
                          self.model.dataset.X_train_bin, self.model.dataset.y_train),
            batch_size=self.train_batch_size,
            shuffle=True if self.model.dataset.seed is None else torch.Generator().manual_seed(self.model.dataset.seed))

        self.val_loader = DataLoader(
            TensorDataset(self.model.dataset.X_val_num, self.model.dataset.X_val_cat, self.model.dataset.X_val_bin,
                          self.model.dataset.y_val),
            batch_size=self.eval_batch_size,
            shuffle=False) if self.model.dataset.y_val is not None else None

        self.test_loader = DataLoader(
            TensorDataset(self.model.dataset.X_test_num, self.model.dataset.X_test_cat, self.model.dataset.X_test_bin,
                          self.model.dataset.y_test),
            batch_size=self.eval_batch_size,
            shuffle=False) if self.model.dataset.y_test is not None else None

        # writer = torch.utils.tensorboard.SummaryWriter(output)  #TODO: add tensorboard? # type: ignore[code]

    def calculate_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(y_pred.shape[1]))

    def train(self, epochs: int, patience: None | int):
        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(patience, mode='max')
        training_log = []

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_num, X_cat, X_bin, y_batch in self.train_loader:
                output = self.model.apply_model(X_num, X_cat, X_bin)
                loss = self.loss_fn(output, y_batch)
                loss.backward()

                # Apply gradient clipping if specified
                if self.gradient_clipping_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)

                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(self.train_loader):.4f}")

    @torch.inference_mode()
    def evaluate(self, mode: str = "Validation"):
        loader = self.val_loader if mode == "Validation" else None
        if loader is None:
            print(f"No {mode.lower()} data provided.")
            return

        self.model.eval()
        total_loss = 0.0
        for X_num, X_cat, X_bin, y_batch in loader:
            output = self.model.apply_model(X_num, X_cat, X_bin)
            loss = self.loss_fn(output, y_batch)
            total_loss += loss.item()

        print(f"{mode} Loss: {total_loss / len(loader):.4f}")
