# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#


import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf

import agents
from common.config import Config
from common.registry import registry


# to register builsers
from datasets.builders import *

# to register processors
from processors import blip_processors

# register
from graphs.models import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config-path", required=True, help="path to configuration file.")
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])


def main():
    args = parse_args()
    config = Config(args)
    setup_seeds(config)

    register_variables()

    agent = agents.setup_agent(config)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
