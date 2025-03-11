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

            if self.config.run.debug_graph_computation:
                self.debug_graph_computation()
                return

            if not self._dataloaders.get("train") and not self.config.run.evaluate:
                raise ValueError("Training dataloader is empty")

            for images, questions, question_ids, img_ids in self._dataloaders:


            if xm.is_master_ordinal():
                if self.config.run.noise_level > 0:
                    xm.master_print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    xm.master_print("No noise will be applied to the image inputs")

            self.load_checkpoint(self._model, self.optimizer, use_cache=self.config.run.use_cache)

        except Exception as e:
            xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")

    def maybe_add_noise(self, batch_sample, noise_level):

        if noise_level > 0:
            image = batch_sample["image"]
            noise = torch.rand_like(image) * noise_level
            batch_sample["image"] = image + noise

    @torch.no_grad()
    def eval(self, epoch):
        running_eval_loss = torch.tensor(0.0, device=self.device)
        total_batches = torch.tensor(0, device=self.device)
        val_loader = self._dataloaders["val"]
        val_loader = pl.MpDeviceLoader(val_loader, self.device)

        if len(val_loader) == 0:
            return float("inf")

        self.model.eval()

        xm.master_print(f"Eval Epoch {epoch} started: {(test_utils.now())}")
        for step, batch_sample in enumerate(val_loader):
            self.maybe_add_noise(batch_sample, self.config.run.noise_level)

            xm.master_print(f"Eval epoch: {epoch}. step: {step} - {(test_utils.now())}")

            with xla_amp.autocast(enabled=self.config.run.amp, device=self.device):
                outputs = self.model(batch_sample)
            loss = outputs["loss"]

            xm.mark_step()

            step_loss = loss.detach()

            running_eval_loss += step_loss
            total_batches += 1

        global_eval_loss = xm.mesh_reduce("running_eval_loss", running_eval_loss.item(), sum)
        global_total_batches = xm.mesh_reduce("total_batches", total_batches.item(), sum)
        eval_avg_loss = global_eval_loss / global_total_batches

        xm.master_print(f"Eval Epoch {epoch} ended: {(test_utils.now())}")
        #self.loss_history["val_loss"].append(eval_avg_loss)        

        return eval_avg_loss

    def debug_graph_computation(self):

        train_loader = self._dataloaders["train"]
        train_loader = pl.MpDeviceLoader(train_loader, self.device)

        self.model.train()

        batch_sample = next(iter(train_loader))
        xm.master_print("Start: debug_graph_computation graph")

        start_epoch = self.load_checkpoint(self._model, self.optimizer)
        if start_epoch > 0:
            self.start_epoch = start_epoch

        with xla_amp.autocast(enabled=self.config.run.amp, device=self.device):
            self.optimizer.zero_grad()
            output = self.model(batch_sample)

        loss = output["loss"]
        loss.backward()
        xm.reduce_gradients(self.optimizer)
        xm.optimizer_step(self.optimizer, barrier=False)
        xm.mark_step()

        xm.master_print(f"Loss value: {loss}")
        self.save_checkpoint_with_optim(self.model, self.optimizer, epoch=self.start_epoch)
        # self.save_checkpoint(self.model, self.optimizer)
        xm.master_print("End: debug_graph_computation graph")

    def validate(self):
        """
        One cycle of model validation
        :return:
        """

    def finalize(self):
        pass

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

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

    def save_checkpoint_with_optim(self, model, optimizer, epoch):

        if xm.is_master_ordinal():

            xm.master_print(f"Saving the checkpoint with optmizer for epoch: {epoch}")

            file_name = self.config.run.resume_ckpt_path
            file_name = f"{file_name}.pth"

            model_state_dict = self.return_state_dict_without_grad(model)
            optimizer_state = self.optimizer_state_without_frozen_params(model, optimizer)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state,
            }

            path = self.config.run.output_dir
            file_and_path = os.path.join(path, file_name)

            xm.master_print(f"Saving Checkpoint in the path: {file_and_path}")
            try:
                self._tpu_metrics.log_checkpoint_saving("Saving checkpoint", epoch=epoch)
                torch.save(checkpoint, file_and_path)
                self._tpu_metrics.log_checkpoint_saving("Checkpoint Saved", epoch=epoch)
                xm.master_print("Checkpoint saved")
            except Exception as e:
                xm.master_print(f"Error saving the checkpoint {e}")

            del checkpoint
            gc.collect()
            xm.mark_step()

        #synchronize all the processes
        #prevent race conditions
        xm.rendezvous("checkpoint_saved")

    def save_checkpoint(self, model, epoch):

        if xm.is_master_ordinal():
            xm.master_print("Saving the checkpoint in the main process")

            model_state_dict = self.return_state_dict_without_grad(model)

            file_name = self.config.run.checkpoint_name
            file_name = f"{file_name}.pth"

            xm.master_print(f"Checkpoint name: {file_name}")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
            }

            path = self.config.run.output_dir
            file_and_path = os.path.join(path, file_name)

            xm.master_print(f"Saving Checkpoint in the path: {file_and_path}")

            torch.save(checkpoint, file_and_path)
            xm.master_print(f"Checkpoint saved at path: {file_and_path}")

        #synchronize all the processes
        #prevent race conditions
        xm.rendezvous("checkpoint_saved")

    def return_state_dict_without_grad(self, model):
        """
        Return the state_dict without the parameters that do not require gradients
        """
        state_dict = {
            k: v.cpu() for k, v in model.state_dict().items()
            if k in [name for name, p in model.named_parameters() if p.requires_grad]
        }

        return state_dict

    def optimizer_state_without_frozen_params(self, model, optimizer):
        """
        Return the optim state_dict without the parameters that do not require gradients
        """
        trainable_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

        filtered_state = {
            k: v for k, v in optimizer.state_dict()['state'].items()
            if k in trainable_param_ids
        }

        state_dict = {
            'state': filtered_state,
            'param_groups': optimizer.state_dict()['param_groups']
        }

        return state_dict

    def prepare_texts(texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(
            conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
