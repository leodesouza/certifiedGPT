import logging
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
from datasets.builders import *  # noqa

# to register processors
from processors import blip_processors  # noqa

# register models
from graphs.models import *  # noqa

# register optimizer and learning rate scheduler
from graphs.models.minigpt4.common.optims import *  # noqa


def parse_args():
    parser = argparse.ArgumentParser(description="Training/Eval/Certification")
    parser.add_argument("mode", choices=["train", "eval", "smoothing_predict", "certify"])
    parser.add_argument("--config-path", required=True, help="path to configuration file.")
    # opcional: args para DDP
    parser.add_argument("--distributed", action="store_true", help="enable torch.distributed (DDP)")
    parser.add_argument("--dist-backend", default="nccl", help="DDP backend (default: nccl)")
    parser.add_argument("--dist-url", default="env://", help="init_method for DDP (default: env://)")
    return parser.parse_args()


def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    out_dir = os.environ.get("OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, "certified.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    registry.register("logger", logger)


def setup_seeds(config):
    seed = getattr(config.run, "seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # melhor desempenho
        torch.backends.cudnn.benchmark = True       # auto-tune convs


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def setup_device_and_rank(args):
    # Single GPU por padrão
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    world_size = 1
    rank = 0
    local_rank = 0
    is_distributed = False

    if args.distributed:
        # torchrun --nproc_per_node=N ... exporta RANK, WORLD_SIZE, LOCAL_RANK
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
            raise RuntimeError("For --distributed, please launch with torchrun or set RANK/WORLD_SIZE/LOCAL_RANK envs.")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend=args.dist-backend if hasattr(args, "dist-backend") else "nccl",
                                             init_method=args.dist_url,
                                             world_size=world_size,
                                             rank=rank)
        is_distributed = True

    return device, rank, local_rank, world_size, is_distributed


def main():
    args = parse_args()
    config = Config(args)

    # Seleciona agente conforme modo
    from agents import BaseAgent, setup_agent  # noqa
    if args.mode == "train":
        print('Running training with agent: minigpt4_finetune_agent')
        from agents import minigpt4_finetune_agent  # noqa
    elif args.mode == "eval":
        print('Running eval with agent: minigpt4_eval_agent')
        from agents import minigpt4_eval_agent  # noqa
    elif args.mode == "smoothing_predict":
        print('Running prediction with agent: minigpt4_predict_agent')
        from agents import minigpt4_predict_agent  # noqa
    elif args.mode == "certify":
        print('Running certifying with agent: minigpt4_certify_agent')
        from agents import minigpt4_certify_agent  # noqa

    setup_logger()
    setup_seeds(config)
    register_variables()

    device, rank, local_rank, world_size, is_distributed = setup_device_and_rank(args)
    registry.register("device", device)
    registry.register("rank", rank)
    registry.register("local_rank", local_rank)
    registry.register("world_size", world_size)
    registry.register("is_distributed", is_distributed)

    # Deixe os agentes lerem 'device' do registry e moverem modelos para CUDA
    agent = setup_agent(config)
    agent.run()
    agent.finalize()

    # Finalização DDP
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
