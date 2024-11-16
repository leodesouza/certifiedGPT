# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

from agents.base import BaseAgent
from common.registry import registry


@registry.register_agent("image_text_finetune")
class ImageTextFinetuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # registry.get_agent_class()

    def evaluation(self, model, dataloader, cuda_enabled=True):
        pass
