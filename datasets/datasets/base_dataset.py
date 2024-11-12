"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):

    def __init__(self, vis_processor=None, text_processor=None, vis_path=None, annotation_path=[]):
        self.vis_root = vis_path
        self.annotations = []

        for ann_path in annotation_path:
            ann = json.load(open(ann_path), "r")
            if isinstance(ann, dict):
                self.annotations.extend(json.load(open(ann_path, "r"))["annotations"])
            else:
                self.annotations.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotations)

    def collater(self, samples):
        return default_collate(samples)

    def set_processor(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor



