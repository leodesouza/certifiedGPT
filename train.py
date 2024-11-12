"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import agents
from common.config import Config


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


def main():
    args = parse_args()
    config = Config(args)
    setup_seeds(config)

    agent = agents.setup_agent(config)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    main()
