"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

from common.registry import registry


class BaseAgent:
    def __init__(self):
        self.datasets = None
        self.config = registry.get_configuration_class("configuration")
        self.logger = logging.getLogger("Agent")

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    def build_datasets(self, config):
        self.datasets = dict()
        datasets_config = config.datasets
        for name in datasets_config:
            builder = registry.get_builder_class(name)
            builder()
            dataset = builder.build_datasets()
            dataset["train"].name = name
            self.datasets["train"] = dataset

        return self.datasets

    def load_checkpoint(self, file_name):
        raise NotImplementedError

    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=0):
        raise NotImplementedError

    def run(self):
        datasets = self.build_datasets(self.config)

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
