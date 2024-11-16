"""
 This file is from
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from common.registry import registry
from configs.all_config_paths import get_database_config_path
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.vqav2_dataset import VQAv2Dataset


@registry.register_builder("vqav2")
class VQAv2Builder(BaseDatasetBuilder):
    train_datasets_cls = VQAv2Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/defaults_vqa.yaml"
    }

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()
        dataset_config = "train"

        datasets[dataset_config] = self.train_datasets_cls(
            vis_processor=self.vis_processor,
            text_processor=self.text_processor
        )

        return datasets


if "__name__" == "__main__":
    datasets = dict()
    vqa_builder = VQAv2Builder()
    dataset = vqa_builder.build_datasets()
    datasets["vqav2"] = dataset
