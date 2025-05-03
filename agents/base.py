import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from common.registry import registry


class BaseAgent:
    def __init__(self):
        self._optimizer = None
        self._lr_scheduler = None
        self._lr_scheduler_plateau = None
        self._model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.config = registry.get("config")
        self._logger = registry.get("logger")
        self.history_file = os.path.join(self.config.output_dir, "history.json")
        self.loss_history = self.load_history() or {"train": [], "val": [], "lr": []}

    def load_checkpoint(self, model, optimizer, use_cache=False):
        ckpt_file = os.path.join(self.config.output_dir, "checkpoint.pt")
        if os.path.isfile(ckpt_file):
            print("Loading checkpoint from:", ckpt_file)
            ckpt = torch.load(ckpt_file, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            return ckpt["epoch"] + 1
        return 0

    def load_finetuned_model(self, model):
        if hasattr(self.config, "load_finetuned") and self.config.load_finetuned:
            ckpt_file = os.path.join(self.config.output_dir, "vqav2.pt")
            if os.path.isfile(ckpt_file):
                print("Loading finetuned model from:", ckpt_file)
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model.load_state_dict(ckpt["model"])

    def save_checkpoint(self, model, optimizer, epoch):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def train(self, epoch):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def model(self):
        return self._model

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls(**kwargs)

    def build_model(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        if self._optimizer is None:
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            self._optimizer = optim.AdamW([
                {"params": decay_params, "weight_decay": 0.01},
                {"params": no_decay_params, "weight_decay": 0.0},
            ], lr=self.config.training.lr, eps=1e-8)
        return self._optimizer

    @property
    def lr_scheduler_plateau(self):
        if self._lr_scheduler_plateau is None:
            self._lr_scheduler_plateau = ReduceLROnPlateau(self.optimizer, mode="min", patience=2, factor=0.5)
        return self._lr_scheduler_plateau

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            num_training_steps = len(self.train_dataloader) * self.config.training.max_epoch
            num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
            self._lr_scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.lr,
                total_steps=num_training_steps,
                pct_start=self.config.training.warmup_ratio,
                anneal_strategy='cos',
                cycle_momentum=False
            )
        return self._lr_scheduler

    @property
    def logger(self):
        return self._logger

    def log_info_master_print(self, message):
        print(message)

    def log_error_master_print(self, message):
        print(message)

    def save_history(self, epoch, train_loss, val_loss, lr):
        self.loss_history["train"].append(train_loss)
        self.loss_history["val"].append(val_loss)
        self.loss_history["lr"].append(lr)

        with open(self.history_file, "w") as f:
            json.dump(self.loss_history, f)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return None

    def plot_result(self, loss_history):
        train_losses = np.array(loss_history['train'])
        val_losses = np.array(loss_history['val'])
        lrs = np.array(loss_history['lr'])

        plt.figure(figsize=(10, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.plot(lrs, label='LR', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training & Validation Loss and Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'loss.png'))
        plt.close()
