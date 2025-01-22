# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



class BaseProcessor:
    def __init__(self, image_size=448, mean=None, std=None):
        self.transform = lambda x: x

        normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),               
                transforms.ToTensor(),
                normalize                
            ]
        )
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, config=None):
        return cls()

    def build(self, **kwargs):
        config = OmegaConf.create(kwargs)

        return self.from_config(config)
