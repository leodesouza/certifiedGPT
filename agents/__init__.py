"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from agents.base import BaseAgent
from agents.image_text_finetune_agent import ImageTextFinetuneAgent
from common.registry import registry


def setup_agent(config):
    assert "agent" in config.run, "Agent name must be provided."

    agent_name = config.run.agent
    agent = registry.get_agent_class(agent_name).setup_agent(cfg=config)
    assert agent is not None, "Agent {} not properly registered.".format(agent_name)

    return agent


__all__ = [
    "BaseAgent",
    "ImageTextFinetuneAgent",
]
