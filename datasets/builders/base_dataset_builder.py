"""
 This file is from
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from tarfile import data_filter

from omegaconf import OmegaConf
import logging

from processors.base_processor import BaseProcessor


def load_dataset_config(config_path):
    config = OmegaConf.load(config_path).datasets
    config = config[list(config.keys())[0]]
    return config


class BaseDatasetBuilder:
    train_datasets_cls, val_datasets_cls, eval_datasets_cls = None, None, None

    def __init__(self, config_path):
        self.config = load_dataset_config(config_path)
        self.vis_processor = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processor = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        logging.info("Building datasets")
        datasets = self.build()
        return datasets

    def build(self):

        self.build_processors()

        build_info = self.config.build_info
        annotations_info = build_info.annotations
        images_info = build_info.images
        datasets = dict()
        for dataset_info in annotations_info.keys():
            if dataset_info not in ["train", "val", "test"]:
                continue

            is_train = dataset_info == "train"

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

            annotation_paths = annotations_info.get(dataset_info).path
            vis_paths = images_info.get(dataset_info).path

            dataset_cls = self.train_datasets_cls if is_train else self.eval_datasets_cls
            datasets[dataset_info] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                annotation_paths=annotation_paths,
                vis_paths=vis_paths
            )

            return datasets

    def build_processors(self):
        pass

    def build_proc_from_config(self, config):
        pass


if __name__ == "__main__":
    config_path = "/home/leonardosouza/projects/certifiedGPT/configs/datasets/vqav2/defaults_vqa.yaml"
    databuilder = BaseDatasetBuilder(config_path)
    databuilder.build_datasets()
