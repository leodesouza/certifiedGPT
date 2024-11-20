# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

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

    @property
    def arch(self):
        return self.config.model.arch

