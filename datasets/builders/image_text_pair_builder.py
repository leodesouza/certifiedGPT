# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

from common.registry import registry
from configs.all_config_paths import get_database_config_path
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from datasets.datasets.vqav2_dataset import VQAv2Dataset, VQAv2TestDataset, VQAv2EvalForCertificationDataset
from datasets.datasets.cc_sbu_align_dataset import CCSbuDataset
from pathlib import Path


@registry.register_builder("vqav2")
class VQAv2Builder(BaseDatasetBuilder):
    train_datasets_cls = VQAv2Dataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/defaults_vqa.yaml"
    }


@registry.register_builder("evalvqav2")
class VQAv2EvalBuilder(BaseDatasetBuilder):    
    
    def __init__(self):
        train_config = registry.get_configuration_class("configuration")
        dataset_config_name = train_config.dataset_config_name if train_config.dataset_config_name else "defaults_vqa.yaml"
        self.eval_datasets_cls = VQAv2Dataset if "eval" in dataset_config_name else VQAv2EvalForCertificationDataset         
        self.DATASET_CONFIG_DICT = {
            "default": f"configs/datasets/vqav2/{dataset_config_name}"        
        }
        super().__init__()    

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
            if dataset_info not in ["val"]:
                continue
            self.logger.info(f"Building dataset: {dataset_info}")

            vis_processor = (
                self.vis_processor["eval"]
            )

            text_processor = (
                self.text_processor["eval"]
            )

            questions_path = questions_info.get(dataset_info).path
            self.logger.info(f"questions_path: {questions_path}")

            annotation_paths = annotations_info.get(dataset_info).path
            self.logger.info(f"annotation_paths: {annotation_paths}")

            vis_paths = Path(images_info.get(dataset_info).path[0])
            self.logger.info(f"vis_paths: {vis_paths}")

            dataset_cls = self.eval_datasets_cls
            datasets[dataset_info] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                questions_paths=questions_path,
                annotation_paths=annotation_paths,
                vis_paths=vis_paths,
                split=dataset_info
            )
        return datasets

    def build_train_processors(self):
        pass

    def build_val_processors(self):
        self.logger.info("Building val processors")
        val_config = registry.get_configuration_class("configuration")

        vis_val_config = val_config.datasets.evalvqav2.vis_processor.val
        text_val_config = val_config.datasets.evalvqav2.text_processor.val

        vis_processor_class = registry.get_processor_class(vis_val_config.name)
        self.logger.info("Building visual processor")
        self.vis_processor["eval"] = vis_processor_class.from_config(vis_val_config)

        text_processor_class = registry.get_processor_class(text_val_config.name)

        self.logger.info("Building textual processor")
        self.text_processor["eval"] = text_processor_class.from_config(text_val_config)



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


@registry.register_builder("testvqav2")
class VQAv2TestBuilder(BaseDatasetBuilder):
    eval_datasets_cls = VQAv2TestDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqav2/eval_vqa_test.yaml"
    }

    def build_train_processors(self):
        pass

    def build_val_processors(self):
        self.logger.info("Building val processors")
        eval_config = registry.get_configuration_class("configuration")

        vis_eval_config = eval_config.datasets.evqav2.vis_processor.test
        text_eval_config = eval_config.datasets.evqav2.text_processor.test

        vis_processor_class = registry.get_processor_class(vis_eval_config.name)
        self.logger.info("Building visual processor")
        self.vis_processor["eval"] = vis_processor_class.from_config(vis_eval_config)

        text_processor_class = registry.get_processor_class(text_eval_config.name)
        self.logger.info("Building textual processor")
        self.text_processor["eval"] = text_processor_class.from_config(text_eval_config)

    def build(self):

        self.build_processors()

        build_info = self.config.build_info
        questions_info = build_info.questions

        images_info = build_info.images
        datasets = dict()
        self.logger.info("Building the dataset based in build options")
        self.logger.info(f"Build path: {self.default_config_path()}")

        for dataset_info in questions_info.keys():
            if dataset_info not in ["train", "val", "test"]:
                continue

            self.logger.info(f"Building eval dataset: {dataset_info}")

            is_train = True if dataset_info in ["train", "val"] else False

            vis_processor = (
                self.vis_processor["train"]
                if is_train
                else self.vis_processor["eval"]
            )

            questions_path = questions_info.get(dataset_info).path
            vis_paths = Path(images_info.get(dataset_info).path[0])

            dataset_cls = self.train_datasets_cls if is_train else self.eval_datasets_cls

            datasets[dataset_info] = dataset_cls(
                questions_path=questions_path,
                vis_processor=vis_processor,
                vis_paths=vis_paths,
                split=dataset_info
            )

        return datasets
