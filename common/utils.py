# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
from common.registry import registry

from torch.utils.data import Dataset
from PIL import Image
import os



def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def load_coco_val2014_annotations():
    import json
    config = registry.get_configuration_class("configuration")
    with open(config.run.coco_annotation_path_file) as f:
        image_objects = json.load(f)
    return image_objects


class FlatImageDatasetWithPaths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path  # Return image and its file path
