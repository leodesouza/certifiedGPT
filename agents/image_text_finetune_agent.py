"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from agents.base import BaseAgent
from common.registry import registry


@registry.register_agent("image_text_finetune")
class ImageTextFinetuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # registry.get_agent_class()

    def evaluation(self, model, dataloader, cuda_enabled=True):
        pass
