import logging
# logging.disable(logging.CRITICAL)
import argparse
import os
import random
import sys

import numpy as np
import torch

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
    parser.add_argument("mode", choices=["train", "eval", "smoothing_predict", "certify"])
    parser.add_argument("--config-path", required=True, help="path to configuration file.")
    args = parser.parse_args()
    return args


def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    log_file_path = os.path.join(os.environ.get("OUTPUT_DIR"),"certified.log")
    file_handler = logging.FileHandler(log_file_path)
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


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def main(rank):
    args = parse_args()
    config = Config(args)    

    from agents import BaseAgent, setup_agent

    if args.mode == "train":
        print('Running training with agent: minigpt4_finetune_agent')
        from agents import minigpt4_finetune_agent
    elif args.mode == "eval":
        print('Running eval with agent: minigpt4_eval_agent')
        from agents import minigpt4_eval_agent
    elif args.mode == "smoothing_predict":
        print('Running training with agent: ??')
        from agents import minigpt4_eval_agent
    elif args.mode == "certify":
        print('Running training with agent: ??')
        from agents import minigpt4_certify_agent

    setup_logger()
    setup_seeds(config)
    register_variables()

    agent = setup_agent(config)
    agent.run()
    agent.finalize()


if __name__ == "__main__":
    
    import torch_xla as xla
    
    _args = parse_args()
    _config = Config(_args)    
    if _config.run.debug_graph_computation:
        print('Running training in debug mode')
        xla.launch(main, args=(), debug_single_process=True)
    else:        
        xla.launch(main, args=())
