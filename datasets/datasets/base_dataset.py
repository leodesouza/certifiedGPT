"""
BSD 3-Clause License

Copyright 2023 Deyao Zhu
All rights reserved.

For full license text, see the LICENSE_MiniGPT-4 file in the repo root or https://github.com/Vision-CAIR/MiniGPT-4/blob/main/LICENSE.md

"""
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]):
        """
        :param vis_processor: visual processor
        :param text_processor: textual processor
        :param vis_root: Root directory of images
        :param ann_paths: directory of annotations file
        """





