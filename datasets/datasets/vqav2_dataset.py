"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
from datasets.datasets.base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_paths, annotation_paths, split="train"):
        super().__init__(
            vis_processor=vis_processor, text_processor=text_processor,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths
        )

        exist_annotation = []
        for annotation in self.annotations:
            image_id = annotation["image_id"]
            file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(self.vis_paths, file_name)
            if os.path.exists(image_path):
                exist_annotation.append(annotation)
        self.annotations = exist_annotation

    def get_data(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        return NotImplementedError


class VQAv2EvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_paths, annotation_paths):
        super().__init__(
            vis_processor=vis_processor, text_processor=text_processor,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths
        )

    def __getitem__(self, index):
        return NotImplementedError
