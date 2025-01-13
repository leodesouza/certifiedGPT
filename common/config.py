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
        
        if hasattr(self.args, 'max_epochs') and int(self.args.max_epochs) > 0:
            self.config.run.max_epochs = int(self.args.max_epochs)  

        if hasattr(self.args, 'batch_size') and int(self.args.batch_size)> 0:
            self.config.datasets.vqav2.batch_size = int(self.args.batch_size)    
        
        if hasattr(self.args, 'checkpoint_name') and self.args.checkpoint_name != "":
            self.config.run.checkpoint_name = self.args.checkpoint_name
        
        if hasattr(self.args, 'num_procs') and int(self.args.num_procs) > 0:
            self.config.run.num_procs = int(self.args.num_procs)
        
        if hasattr(self.args, 'noise_level') and int(self.args.noise_level) > 0:
            self.config.config.datasets.vqav2.noise_level = int(self.args.noise_level)                                                
        
    @property
    def datasets(self):
        return self.config.datasets

    @property
    def run(self):
        return self.config.run

    @property
    def arch(self):
        return self.config.model.arch

    @property
    def model(self):
        return self.config.model
