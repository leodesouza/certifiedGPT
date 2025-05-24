import logging
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Local imports
from common.config import Config
from common.registry import registry
from datasets.builders import *
from processors import blip_processors
from graphs.models import *
from graphs.models.minigpt4.common.optims import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("mode", choices=["train", "eval", "smoothing_predict", "certify", "transfer_based_attack", "img_t2_text", "chat"])
    parser.add_argument("--config-path", required=True, help="Path to configuration file.")
    return parser.parse_args()


def setup_logger(rank):
    logger = logging.getLogger(f"logger_rank_{rank}")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    log_file_path = os.path.join(os.environ.get("OUTPUT_DIR", "."), f"certified_rank{rank}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    registry.register("logger", logger)


def setup_seeds(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])


def init_distributed(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    try:
        init_distributed(rank, world_size)

        config = Config(args)
        setup_seeds(config.run.seed, rank)
        setup_logger(rank)
        register_variables()

        device = torch.device(f"cuda:{rank}")
        registry.register("device", device)
        registry.register("rank", rank)
        registry.register("world_size", world_size)

        # Import after rank/device setup
        from agents import BaseAgent, setup_agent

        if args.mode == "train":
            print(f"[Rank {rank}] Running training with agent: minigpt4_finetune_agent")
            from agents import minigpt4_finetune_agent
        elif args.mode == "eval":
            print(f"[Rank {rank}] Running eval with agent: minigpt4_eval_agent")
            from agents import minigpt4_eval_agent
        elif args.mode == "smoothing_predict":
            print(f"[Rank {rank}] Running predict with agent: minigpt4_predict_agent")
            from agents import minigpt4_predict_agent
        elif args.mode == "certify":
            print(f"[Rank {rank}] Running certifying with agent: minigpt4_certify_agent")
            from agents import minigpt4_certify_agent
        elif args.mode == "transfer_based_attack":
            print(f"[Rank {rank}] Running transfer based attacks..")            
            sys.argv = ["_train_adv_img_trans.py"]
            from  experiments.attacks._train_adv_img_trans import main
            main()
        elif args.mode == "img_t2_text":
            print(f"[Rank {rank}] Running img_t2_text from MiniGPT4.")            
            sys.argv = ["_minigpt4_img2txt.py"]
            from  experiments.attacks._minigpt4_img2txt import main
            main()
        elif args.mode == "query_based_attack":
            print(f"[Rank {rank}] Running query_based_attack from MiniGPT4.")            
            sys.argv = ["_train_adv_img_query.py"]
            from  experiments.attacks._train_adv_img_query import main
            main()
        elif args.mode == "chat":
            print(f"[Rank {rank}] Running chat from MiniGPT4.")            
            sys.argv = ["demo_chat.py"]
            from graphs.models.minigpt4.demo_chat import main
            main()

            
        
        if args.mode not in ["transfer_based_attack", "query_based_attack", "img_t2_text", "chat"]:             
            agent = setup_agent(config)
            agent.run()
            agent.finalize()

    except Exception as e:
        logger = logging.getLogger(f"logger_rank_{rank}")
        logger.error(f"Exception in rank {rank}: {e}", exc_info=True)
    finally:
        cleanup_distributed()


def launch_distributed():
    args = parse_args()
    world_size = torch.cuda.device_count()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)


if __name__ == "__main__":
    launch_distributed()
