# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
import time
from datetime import datetime

#Torch 
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

#Pytorch XLA

from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.amp as xla_amp

from utils.gcsfuse import mount_gcsfuse
import wandb
from tqdm import tqdm

from agents.base import BaseAgent
from common.metrics import TPUMetrics
from common.registry import registry
import torch_xla.test.test_utils as test_utils
from graphs.models.minigpt4.conversation.conversation import CONV_VISION_minigptv2

import gc

# rank and world size are inferred from XLA Device
# source: https://github.com/pytorch/xla/
dist.init_process_group(backend='xla', init_method='xla://')


@registry.register_agent("image_text_eval")
class MiniGPT4EvalAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.start_step = 0
        self.max_epoch = self.config.run.max_epoch
        self._device = xm.xla_device()
        self._model = None  # self.build_model()
        # self._setup_wandb(self._model)
        self._start_epoch = 0
        self._tpu_metrics = TPUMetrics()

    def run(self):
        try:

            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if xm.is_master_ordinal():
                if self.config.run.noise_level > 0:
                    xm.master_print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    xm.master_print("No noise will be applied to the image inputs")

            self.load_checkpoint(self._model, self.optimizer, use_cache=self.config.run.use_cache)

            self.eval(self._dataloaders)

        except Exception as e:
            xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")

    @torch.no_grad()
    def eval(self, dataloader):
        running_eval_loss = torch.tensor(0.0, device=self.device)
        total_batches = torch.tensor(0, device=self.device)
        val_loader = dataloader["val"]
        val_loader = pl.MpDeviceLoader(val_loader, self.device)

        if len(val_loader) == 0:
            return float("inf")

        conv_temp = CONV_VISION_minigptv2.copy()
        conv_temp.system = ""

        xm.master_print(f"Eval started: {(test_utils.now())}")
        predictions = []
        self.model.eval()
        for step, batch_sample in enumerate(val_loader):
            xm.master_print(f"Eval step: {step} - {(test_utils.now())}")

            self.maybe_add_noise(batch_sample, self.config.run.noise_level)
            image = batch_sample["image"]
            questions = batch_sample["question"]
            question_ids = batch_sample["question_id"]
            img_ids = batch_sample["img_id"]

            texts = self.prepare_texts(question, conv_temp)
            answers = self.model.generate(image, texts, max_new_tokens=self.config.run.max_new_tokens, do_sample=False)
            xm.mark_step()

            for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
                result = ()
                answer = answer.lower().replace('<unk>','').strip()
                result['answer'] = answer
                result['question_id'] = int(question_id)
                predictions.append(result)



        global_eval_loss = xm.mesh_reduce("running_eval_loss", running_eval_loss.item(), sum)
        global_total_batches = xm.mesh_reduce("total_batches", total_batches.item(), sum)
        eval_avg_loss = global_eval_loss / global_total_batches

        xm.master_print(f"Eval ended: {(test_utils.now())}")

        return eval_avg_loss

    def finalize(self):
        pass

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    @staticmethod
    def maybe_add_noise(batch_sample, noise_level):

        if noise_level > 0:
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
                        "Loaded {} records for split {}".format(num_records, dataset)
                    )

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

    def create_optimizer(self):
        self.logger.info("Creating the optimizer")
        beta1 = self.config.run.beta1
        beta2 = self.config.run.beta2

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.run.init_lr),
            weight_decay=float(self.config.run.weight_decay),
            betas=(beta1, beta2),
        )

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        model.to(self.device)
        return model

    def prepare_texts(texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(
            conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
