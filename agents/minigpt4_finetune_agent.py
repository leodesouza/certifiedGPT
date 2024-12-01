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
        self.start_epoch = 1
        self.max_epoch = self.config.run.max_epoch
        self._model = self.build_model()
        self._device = None
        self._start_epoch = 0
        self._optimizer = self.create_optimizer()
        self._dataloaders = None
        self._scaler = None

    def run(self):
        start_time = time.time()
        best_epoch = 0
        running_training_loss = 0
        running_eval_loss = 0

        if not self.config.run.evaluate_only and self.config.run.resume_ckpt_path is not None:
            logging.info(f"Loading the checkpoint from path: {self.config.run.resume_ckpt_path}")
            self.load_checkpoint(self.config.run.resume_ckpt_path)

        logging.info(f"Set model to device: {self.config.run.device}")
        self.model = self.model.to(self.config.run.device)

        logging.info("Creating the dataloaders")
        self._dataloaders = self.create_dataloaders()

        logging.info("Start running the training loop")
        logging.info(f"Start epoch: {self.start_epoch}. Max epoch: {self.max_epoch}")
        for epoch in range(self.start_epoch, self.max_epoch):
            # training step
            if not self.config.run.evaluate_only:
                logging.info("Start training")
                self.train(epoch)

            # evaluation step
            self.eval(epoch)

        logging.info(f"Finished the training loop")

    def train(self, epoch):
        train_loader = self._dataloaders["train"]
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

    @torch.no_grad()
    def eval(self, epoch):
        running_eval_loss = 0
        val_loader = self._dataloaders["val"]
        self.model.eval()

        if not hasattr(val_loader, "__next__"):
            val_loader = iter(val_loader)

        samples = next(val_loader)
        samples = samples.to(self.config.run.device)

        with torch.cuda.amp.autocast(enabled=self.config.run.amp):
            loss = self.model(samples)['loss']
            running_eval_loss += loss.item()

        avg_valid_loss = running_eval_loss / len(val_loader)
        # print(f"Epoch [{epoch+1}/{}]")

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
        logging.info("Creating the optimizer")
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

    def build_model(self):
        logging.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        return model
