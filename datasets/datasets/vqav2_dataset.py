"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from datasets.datasets.base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)
    
    