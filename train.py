import logging
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch_xla.distributed.xla_multiprocessing as xmp
from omegaconf import OmegaConf


# local imports 

from common.config import Config
from common.registry import registry

# to register builders
from datasets.builders import *

# to register processors
from processors import blip_processors

# register models
from graphs.models import *

# register optimizer and learning rate scheduler
from graphs.models.minigpt4.common.optims import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config-path", required=True, help="path to configuration file.")    
    parser.add_argument("--noise_level", required=True, help="noise level to apply on images.")    
    parser.add_argument("--num_procs", required=True, help="num of tpu cores.")    
    parser.add_argument("--max_epochs", required=True, help="max epochs to train.")    
    parser.add_argument("--batch_size", required=True, help="batch size.")    
    parser.add_argument("--checkpoint_name", required=True, help="checkpoint name.")    
    
    
    args = parser.parse_args()

    return args


def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler('certifiedgpt.log')
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    registry.register("logger", logger)


def setup_seeds(config):
    seed = config.run.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])

def main(index):
    
    import agents
    
    setup_logger()
    args = parse_args()
    config = Config(args)
    setup_seeds(config)    
    register_variables()    

    agent = agents.setup_agent(config)
    agent.run()
    
            
if __name__ == "__main__":
    
    import torch_xla as xla 
    
    xla.launch(main, args=())   