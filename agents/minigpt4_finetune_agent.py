# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
import logging
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler

from agents.base import BaseAgent
from common.registry import registry


@registry.register_agent("image_text_finetune")
class MiniGPT4FineTuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.max_epoch = self.config.run.max_epoch
        self._model = None  # registry.get_model_class(self.config.arch)
        self._device = None
        self._start_epoch = 0
        self._optimizer = self.create_optimizer()
        self._scaler = None

    def run(self):
        start_time = time.time()
        best_epoch = 0

        # resume from checkpoint...
        # if not config.run.evaluate_only..

        self.model = self.model.to(self.config.run.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            # training step
            if not self.config.run.evaluate:
                logging.info("Start training")
                self.train(epoch)

            # evaluation step


        # datasets = self.build_datasets()

    def train(self, epoch):

        dataloaders = self.create_dataloaders()
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        self.model.train()
        self._optimizer.zero_grad()

        if not hasattr(train_loader, "__next__"):
            train_loader = iter(train_loader)

        samples = next(train_loader)
        samples = samples.to(self.config.run.device)

        with torch.cuda.amp.autocast(enabled=self.config.run.amp):
            loss = self.model(samples)['loss']

        if self.config.run.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.config.run.amp:
            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            self._optimizer.step()

    def train_one_epoch(self):
        """
        Execute only one training loop
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """

    def finalize(self):
        """
        Finalize all operations and dataloaders
        :return:
        """
        raise NotImplementedError

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run.device)
        return self._device

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

    def create_dataloaders(self):
        logging.info("building datasets")
        datasets = self._build_datasets()
        dataset_names = sorted(datasets.keys())
        dataloaders = dict()

        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]

            for split in dataset.values():
                num_records = len(split)
                if num_records >= 0:
                    logging.info("Loaded {} records for split {}".format(num_records, dataset))

                is_train = True if split.split_name in self.config.run.train_splits else False

                collate_fn = getattr(split, "collater", None)

                loader = DataLoader(
                    split,
                    batch_size=self.config.datasets[dataset_name].batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=True if is_train else False,
                    collate_fn=collate_fn
                )
                dataloaders[split.split_name] = loader

        return dataloaders

    def create_optimizer(self):

        # optim_params = [
        #     {
        #         "params"
        #     }
        # ]
        beta1 = self.config.run.beta1
        beta2 = self.config.run.beta2

        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.run.init_lr),
            weight_decay=float(self.config.run),
            betas=(beta1, beta2)
        )
