# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "Pytorch-Project-Template" project.
# See LICENSE_Template.md for the full license text or visit the repo at:
# https://github.com/moemen95/Pytorch-Project-Template/
#

import logging
import torch
import torch_xla.core.xla_model as xm
import common
from common.registry import registry
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils


class BaseAgent:
    def __init__(self):
        self.lr_sched = None
        self.lr_sched_plateau = None
        self._optimizer = None
        self._scaler = None
        self._model = None
        self._device = None
        self.config = registry.get_configuration_class("configuration")
        self._dataloaders = None

    def load_checkpoint(self, file_name):
        """
        Latest/saved checkpoint
        :param file_name:
        :return:
        """        
        raise NotImplementedError

    def save_checkpoint(self, model, optimizer, epoch, loss, file_name="checkpoint.pth.bar", is_best=False):
        """

        :param file_name:  name of the checkpoint to save
        :param is_best:  bool indicating if the checkpoint is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """

    def train(self, epoch):
        """
        train a specific epoch
        :return:
        """

    def train_one_epoch(self):
        """
        Execute only one training loop
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """

    def finalize(self):
        """
        Finalize all operations and dataloaders
        :return:
        """
        raise NotImplementedError

    @property
    def device(self):                    
        return self._device

    @property
    def model(self):                
        return self._model

    # @property
    # def scaler(self):
    #     amp = self.config.run.amp
    #     if amp:
    #         self._scaler = torch.amp.GradScaler()
    #     return self.scaler

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    def build_model(self):
        """
        Return the current instance of model to train
        :return: ModelBase
        """
        return NotImplementedError()

    @property
    def optimizer(self):
        if self._optimizer is None:            
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():                                

                if not p.requires_grad:
                    continue  # frozen weights                
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            #self.logger.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run.init_lr),
                weight_decay=float(self.config.run.weight_decay),
                betas=(0.9, beta2),
                foreach=False
            )

        return self._optimizer

    @property
    def lr_scheduler_plateau(self):

        if self.lr_sched_plateau is None:
            self.lr_sched_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1, # reduce LR by 10x
                patience=3,  # epochs that scheduler will wait to check and aplly lr reducing
                threshold=0.0001, # minimum change to qualify as "improvement"
                cooldown=1, # epochs that scheduler will pause lr checkings
                min_lr=self.config.run.min_lr
            )

        return self.lr_sched_plateau

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self.lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run.lr_sched)

            max_epoch = self.config.run.max_epoch
            min_lr = self.config.run.min_lr
            init_lr = self.config.run.init_lr

            # optional parameters
            decay_rate = self.config.run.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run.get("warmup_start_lr", -1)
            warmup_steps = self.config.run.get("warmup_steps", 0)
            warmup_max_lr = self.config.run.get("warmup_max_lr", 0)
            iters_per_epoch = self.config.run.get("iters_per_epoch", None)                                    

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self._dataloaders["train"])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self.lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
                warmup_max_lr=warmup_max_lr
            )

        return self.lr_sched

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger            
