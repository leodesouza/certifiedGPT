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
from datasets.datasets.cc_sbu_align_dataset import CCSbuDataset
from pathlib import Path


@registry.register_builder("vqav2")
class VQAv2Builder(BaseDatasetBuilder):
    train_datasets_cls = VQAv2Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/defaults_vqa.yaml"
    }

@registry.register_builder("cc_sbu")
class CCSbuBuilder(BaseDatasetBuilder):
    train_datasets_cls = CCSbuDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/defaults.yaml"
    }   

    def build(self):

        self.build_processors()
        build_info = self.config.build_info        
        annotations_info = build_info.annotations

        images_info = build_info.images
        datasets = dict()
        self.logger.info("Building the dataset based in build options")
        self.logger.info(f"Build path: {self.default_config_path()}")

        vis_processor = self.vis_processor["train"]
        text_processor = self.text_processor["train"]

        
        annotation_paths = Path(annotations_info.get('train').path[0])
        vis_paths = Path(images_info.get('train').path[0])

        dataset_cls = self.train_datasets_cls
        datasets['train'] = dataset_cls(
            vis_processor=vis_processor,
            text_processor=text_processor,
            annotation_paths=annotation_paths,
            vis_paths=vis_paths,
            split='train'
        )
                
        return datasets
    
    def build_processors(self):
       self.build_train_processors()
       
    def build_train_processors(self):
        self.logger.info("Building val processors")
        train_config = registry.get_configuration_class("configuration")

        vis_train_config = train_config.datasets.cc_sbu.vis_processor.train
        text_train_config = train_config.datasets.cc_sbu.text_processor.train

        vis_processor_class = registry.get_processor_class(vis_train_config.name)
        self.logger.info("Building visual processor")
        self.vis_processor["train"] = vis_processor_class.from_config(vis_train_config)

        text_processor_class = registry.get_processor_class(text_train_config.name)

        self.logger.info("Building textual processor")
        self.text_processor["train"] = text_processor_class.from_config(text_train_config)        

    


