"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import torch

from common.registry import registry


class BaseAgent:
    def __init__(self):
        self.logger = logging.getLogger("Agent")
        self.datasets = None
        self.config = registry.get_configuration_class("configuration")
        self._model = registry.get_model_class(self.config.arch)
        self._device = None
        self._dataloaders = None
        self._start_epoch = 0
        self._scaler = None
        # self.setup_output_dir()

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
        amp = self.config.run.get("amp", False)
        if amp:
            self._scaler = torch.cuda.amp.GradScaler()
        return self.scaler

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    def load_checkpoint(self, file_name):
        raise NotImplementedError

    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=0):
        raise NotImplementedError

    def run(self):
        datasets = self.build_datasets()

    def build_datasets(self):
        self.datasets = dict()
        datasets_config = self.config.datasets
        for name in datasets_config:
            builder = registry.get_builder_class(name)
            builder_instance = builder()
            dataset = builder_instance.build_datasets()
            self.datasets = dataset

        return self.datasets

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
