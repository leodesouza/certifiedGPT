# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import json
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from common.registry import  registry


class BaseDataset(Dataset):

    def __init__(self,
                 vis_processor=None, text_processor=None,
                 questions_paths=[],
                 vis_paths=None, annotation_paths=[],
                 split="train"):

        self.vis_paths = vis_paths

        self.questions = []

        self.logger.info("Loading dataset ...")
        self.logger.info("Loading questions json files")
        for question_path in questions_paths:
            question = json.load(open(question_path, "r"))
            if isinstance(question, dict):
                self.questions.extend(json.load(open(question_path, "r"))["questions"])

        self.annotations = []
        self.images_dict = {}
        self.questions_dict = {}

        self.logger.info("Loading annotations json files")
        for ann_path in annotation_paths:
            ann = json.load(open(ann_path, "r"))
            if isinstance(ann, dict):
                self.annotations.extend(json.load(open(ann_path, "r"))["annotations"])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger

    def __len__(self):
        return len(self.questions)

    def collater(self, samples):
        return default_collate(samples)

    def set_processor(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
