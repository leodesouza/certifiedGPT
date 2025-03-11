# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#


from agents.base import BaseAgent
#from agents.minigpt4_finetune_agent import MiniGPT4FineTuneAgent
from common.registry import registry


def setup_agent(config):
    assert "agent" in config.run, "Agent name must be provided."

    agent_name = config.run.agent
    agent = registry.get_agent_class(agent_name).setup_agent(cfg=config)
    assert agent is not None, "Agent {} not properly registered.".format(agent_name)

    return agent


__all__ = [
    "BaseAgent"
]
