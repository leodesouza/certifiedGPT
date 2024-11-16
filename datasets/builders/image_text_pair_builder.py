# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

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


