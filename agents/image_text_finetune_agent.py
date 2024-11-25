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
class ImageTextFinetuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.max_epoch = self.config.run.max_epoch
        self._model = None  # registry.get_model_class(self.config.arch)
        self._device = None
        self._dataloaders = dict()
        self._start_epoch = 0
        self._scaler = None

    def run(self):
        start_time = time.time()
        best_epoch = 0

        # resume from checkpoint...
        # if not config.run.evaluate_only..
        self.create_dataloaders()

        for epoch in range(self.start_epoch, self.max_epoch):
            if not self.config.run.evaluate:
                logging.info("Start training")
                train_stats = self.train_epoch(epoch)

        # datasets = self.build_datasets()

    def train(self):
        """
        Main training loop
        :return:
        """

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

        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]
            loaders = []
            for split in dataset.values():
                num_records = len(split)
                if num_records >= 0:
                    logging.info("Loaded {} records for split {}".format(num_records, dataset))

                is_train = True if split.split_name in self.config.run.train_splits else False

                collate_fn = getattr(split, "collater", None)

                loader = DataLoader(
                    split,
                    batch_size=self.config.datasets.vqav2.batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=True if is_train else False,
                    collate_fn=collate_fn
                )
                loaders.append(loader)

            self._dataloaders[dataset_name] = loaders

            for dataloader in self._dataloaders.values():
                for loader in dataloader:
                    for batch in loader:
                        break

