# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "Pytorch-Project-Template" project.
# See LICENSE_Template.md for the full license text or visit the repo at:
# https://github.com/moemen95/Pytorch-Project-Template/
#

import logging
import torch

import common
from common.registry import registry


class BaseAgent:
    def __init__(self):
        self._scaler = None
        self._model = None
        self._device = None
        self.config = registry.get_configuration_class("configuration")

    def load_checkpoint(self, file_name):
        """
        Latest/saved checkpoint
        :param file_name:
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.bar", is_best=0):
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
        if self._device is None:
            self._device = torch.device(self.config.run.device)
        return self._device

    @property
    def model(self):
        if self._model.device != self._device:
            self._model = self._model.to(self.device)
        return self._model

    @property
    def scaler(self):
        amp = self.config.run.amp
        if amp:
            self._scaler = torch.amp.GradScaler()
        return self.scaler

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
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger



