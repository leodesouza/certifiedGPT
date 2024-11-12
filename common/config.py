"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from omegaconf import OmegaConf

from common.registry import registry


class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args
        # Register the config and configuration for setup
        registry.register("configuration", self)
        self.config = OmegaConf.load(self.args.config_path)

    @property
    def datasets(self):
        return self.config.datasets

    @property
    def run(self):
        return self.config.run

