import os
import sys

from omegaconf import OmegaConf
from common.registry import registry

from common import utils
from configs.all_config_paths import get_database_config_path, DATASET_CONFIG_DICT


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])


def default_config_path(key="vqav2"):
    return utils.get_abs_path(DATASET_CONFIG_DICT[key])


def load_dataset_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    config_datasets = config.datasets
    config_datasets = config_datasets[list(config_datasets.keys())[0]]
    return config_datasets


def generate():
    config_file_path = os.environ["CONFIG_VQA_FILE"]
    config = load_dataset_config(config_file_path)
    path = get_database_config_path("vqav2")


if __name__ == "__main__":
    register_variables()
    generate()
