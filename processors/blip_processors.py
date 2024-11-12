"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import re

from omegaconf import OmegaConf
from torchvision.transforms.functional import InterpolationMode

from common.registry import registry
from processors.base_processor import BaseProcessor
from torchvision import transforms


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        # if mean is None:
        #     mean = (0.48145466, 0.4578275, 0.40821073)
        # if std is None:
        #     std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_zise=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_zise, image_zise),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize
            ]
        )

        def __call__(self, item):
            return transforms(item)

        @classmethod
        def from_config(cls, config=None):
            image_size = config.get("imagem_size", 224)

            mean = config.get("mean", None)
            std = config.get("std", None)
            min_scale = config.get("min_scale", None)
            max_scale = config.get("max_scale", None)

            return cls(
                image_size=image_size,
                mean=mean,
                std=std,
                min_scale=min_scale,
                max_scale=max_scale
            )


class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_zise=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_zise, image_zise),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize
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

            return cls(
                image_size=image_size,
                mean=mean,
                std=std
            )


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self,prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

    @classmethod
    def from_config(cls, config=None):
        if config is None:
            config = OmegaConf.create()
        prompt = config.get("prompt", "")
        max_words = config.get("max_words", "")
        return cls(prompt=prompt, max_words=prompt)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption
