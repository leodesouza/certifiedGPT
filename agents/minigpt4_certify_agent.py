# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
# Torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


# Pytorch XLA

from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from agents.base import BaseAgent
from common.metrics import TPUMetrics
from common.registry import registry
import torch_xla.test.test_utils as test_utils
from randomized_smoothing.smoothing import Smooth
from bert_score import score


# rank and world size are inferred from XLA Device
# source: https://github.com/pytorch/xla/
dist.init_process_group(backend='xla', init_method='xla://')


@registry.register_agent("image_text_eval")
class MiniGPT4CertifyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_step = 0
        self._device = xm.xla_device()
        self._model = None # self.build_model()
        self._tpu_metrics = TPUMetrics()
        self.questions_paths = None
        self.annotations_paths = None
        self.smoothed_decoder = Smooth(self._model, 0, self.config.run.noise_level)

    def run(self):
        try:

            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if xm.is_master_ordinal():
                if self.config.run.noise_level > 0:
                    xm.master_print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    xm.master_print("No noise will be applied to the image inputs")

            # self.load_finetuned_model(self._model)
            self.certify(self._dataloaders)

        except Exception as e:
            xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")
            self.logger.error(f"Error on agent run: {test_utils.now()}. Details: {e}")

    @torch.no_grad()
    def certify(self, dataloader):
        total_batches = torch.tensor(0, device=self.device)
        val_loader = dataloader["val"]
        # val_loader = pl.MpDeviceLoader(val_loader, self.device)
        n0 = self.config.run.number_monte_carlo_samples_for_selection
        n = self.config.run.number_monte_carlo_samples_for_estimation

        if len(val_loader) == 0:
            return float("inf")

        xm.master_print(f"Certify started: {(test_utils.now())}")
        predictions = []

        # self.model.eval()
        for step, batch_sample in enumerate(val_loader):
            # certify every skip examples and break when step == max
            if step % self.config.run.skip != 0:
                continue
            if step == self.config.run.max:
                break

            xm.master_print(f"Certify step: {step} - {(test_utils.now())}")

            image = batch_sample["image"]
            answer = batch_sample["answer"]

            # certify prediction of smoothed decoder around images
            prediction, radius = self.smoothed_decoder.certify(image, n0, n, self.config.run.alpha, batch_size=self.config.datasets.evalvqav2.batch_size)
            is_similar = prediction == answer
            total_batches += 1

        xm.master_print(f"Certify ended: {(test_utils.now())}")


    def compute_bertscore(predictions, answers, lang='en'):
        p, r, f1 = score(predictions, answers, lang=lang, rescale_with_baseline=True)

        return {
            "precision": p.mean().item(),
            "recall": r.mean().item(),
            "f1": f1.mean().item()
        }

    def finalize(self):
        pass

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    @staticmethod
    def add_noise(batch_sample, noise_level):
        image = batch_sample["image"]
        noise = torch.rand_like(image) * noise_level
        batch_sample["image"] = image + noise

    def _build_datasets(self):
        datasets = dict()
        datasets_config = self.config.datasets
        for name in datasets_config:
            builder = registry.get_builder_class(name)
            builder_instance = builder()
            dataset = builder_instance.build_datasets()
            datasets[name] = dataset

        return datasets

    def create_dataloaders(self, batch_size=-1):
        self.logger.info("building datasets")
        datasets = self._build_datasets()
        dataset_names = sorted(datasets.keys())
        dataloaders = dict()

        for dataset_name in dataset_names:

            dataset = datasets[dataset_name]

            for split in dataset.values():
                num_records = len(split)
                if num_records >= 0:
                    self.logger.info(
                        "Loaded {} records for split {}".format(num_records, split.split_name)
                    )

                self.questions_paths = split.questions_paths
                self.annotations_paths = split.annotations_paths

                is_train = (
                    True if split.split_name in self.config.run.train_splits else False
                )

                collate_fn = getattr(split, "collater", None)

                sampler = DistributedSampler(
                    split,
                    num_replicas=xr.world_size(),
                    rank=xm.runtime.global_ordinal(),
                    shuffle=True if is_train else False
                ) if self.config.run.distributed and xr.world_size() > 1 else None

                loader = DataLoader(
                    split,
                    batch_size=batch_size if batch_size > 0 else self.config.datasets[dataset_name].batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=(True if is_train and not self.config.run.distributed else False),
                    sampler=sampler,
                    collate_fn=collate_fn,
                    drop_last=True
                )
                dataloaders[split.split_name] = loader

        return dataloaders

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        model.to(self.device)
        return model

    def prepare_texts(texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(
            conv.roles[0], text) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts


