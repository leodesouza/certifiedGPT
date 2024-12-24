# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import re
import ast

from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode

from common.registry import registry
from processors.base_processor import BaseProcessor
from torchvision import transforms


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BaseProcessor):
    def __init__(
        self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__()

        normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                # transforms.RandomResizedCrop(
                #     (image_zise, image_zise),
                #     (min_scale, max_scale),
                #     interpolation=InterpolationMode.BICUBIC
                # ),
                transforms.ToTensor(),
                normalize
                # transforms.Lambda(
                #     lambda x: (x - x.min()) / (x.max() - x.min() + 1e-7)
                # ),  # Min-max scaling
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, config=None):
        image_size = config.get("imagem_size", 224)

        mean = config.get("mean", None)
        mean = ast.literal_eval(mean)

        std = config.get("std", None)
        std = ast.literal_eval(std)
        min_scale = config.get("min_scale", None)
        max_scale = config.get("max_scale", None)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


class Blip2ImageEvalProcessor(BaseProcessor):
    def __init__(
        self, image_zise=224, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_zise, image_zise), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        def __call__(self, item):
            return transforms(item)

        @classmethod
        def from_config(cls, config=None):
            image_size = config.get("imagem_size", 224)

            mean = config.get("mean", None)
            std = config.get("std", None)
            # min_scale = config.get("min_scale", None)
            # max_scale = config.get("max_scale", None)

            return cls(image_size=image_size, mean=mean, std=std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)
        return caption

    @classmethod
    def from_config(cls, config=None):
        if config is None:
            config = OmegaConf.create()
        prompt = config.get("prompt", "")
        max_words = config.get("max_words", 100)
        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        # replaces characters .!\"()*#:;~ by " "
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )

        # replace multiple spaces by " "
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )

        # remove trailling spaces
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption
