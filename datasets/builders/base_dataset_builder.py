# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
from pathlib import Path

from omegaconf import OmegaConf
import common
from common import utils
from processors.base_processor import BaseProcessor
from common.registry import registry

import os

# Register a resolver for the `env` type
OmegaConf.register_new_resolver("env", lambda key: os.environ.get(key, None), replace=True)


# OmegaConf.register_resolver("env", lambda key: os.environ.get(key, None))


def load_dataset_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    config_datasets = config.datasets
    config_datasets = config_datasets[list(config_datasets.keys())[0]]
    return config_datasets


class BaseDatasetBuilder:
    train_datasets_cls, val_datasets_cls, eval_datasets_cls = None, None, None

    def __init__(self):
        self.config = load_dataset_config(self.default_config_path())
        self.vis_processor = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processor = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        self.logger.info("Building datasets")
        datasets = self.build()
        return datasets

    def build(self):

        self.build_processors()

        build_info = self.config.build_info
        questions_info = build_info.questions
        annotations_info = build_info.annotations

        images_info = build_info.images
        datasets = dict()
        self.logger.info("Building the dataset based in build options")
        self.logger.info(f"Build path: {self.default_config_path()}")

        for dataset_info in annotations_info.keys():
            if dataset_info not in ["train", "val"]:
                continue
            self.logger.info(f"Building dataset: {dataset_info}")

            is_train = True if dataset_info in ["train", "val"] else False

            vis_processor = (
                self.vis_processor["train"]
                if is_train
                else self.vis_processor["eval"]
            )

            text_processor = (
                self.text_processor["train"]
                if is_train
                else self.text_processor["eval"]
            )

            questions_path = questions_info.get(dataset_info).path
            annotation_paths = annotations_info.get(dataset_info).path
            vis_paths = Path(images_info.get(dataset_info).path[0])

            dataset_cls = self.train_datasets_cls if is_train else self.eval_datasets_cls

            datasets[dataset_info] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                questions_paths=questions_path,
                annotation_paths=annotation_paths,
                vis_paths=vis_paths,
                split=dataset_info
            )
        return datasets

    def build_processors(self):
        self.build_train_processors()
        self.build_val_processors()

    def build_train_processors(self):
        self.logger.info("Building val processors")

        train_config = registry.get_configuration_class("configuration")
        vis_train_config = train_config.datasets.vqav2.vis_processor.train

        vis_processor_class = registry.get_processor_class(vis_train_config.name)
        self.logger.info("Building visual processor")
        self.vis_processor["train"] = vis_processor_class.from_config(vis_train_config)

    def build_val_processors(self):
        self.logger.info("Building val processors")
        val_config = registry.get_configuration_class("configuration")

        vis_val_config = val_config.datasets.vqav2.vis_processor.val
        text_val_config = val_config.datasets.vqav2.text_processor.val

        vis_processor_class = registry.get_processor_class(vis_val_config.name)
        self.logger.info("Building visual processor")
        self.vis_processor["val"] = vis_processor_class.from_config(vis_val_config)

        text_processor_class = registry.get_processor_class(text_val_config.name)

        self.logger.info("Building textual processor")
        self.text_processor["val"] = text_processor_class.from_config(text_val_config)

    def default_config_path(self, key="default"):
        return utils.get_abs_path(self.DATASET_CONFIG_DICT[key])

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger
