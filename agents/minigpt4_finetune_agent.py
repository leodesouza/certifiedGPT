# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
import os 
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler

from agents.base import BaseAgent
from common.registry import registry
from graphs.losses.cross_entropy_loss import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt


torch.autograd.set_detect_anomaly(False)


def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()

    for loss_name, loss_values in losses.items():
        ax.plot(loss_values, label=loss_name)

    ax.legend(fontsize=16)
    ax.set_xlabel("Iteration", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title("Loss vs iterations", fontsize=16)
    plt.savefig("vqa_plot_training.png")
    plt.close()


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)
    return samples


@registry.register_agent("image_text_finetune")
class MiniGPT4FineTuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.max_epoch = self.config.run.max_epoch
        self._model = self.build_model()
        self._start_epoch = 0
        self._scaler = None

    def run(self):
        start_time = time.time()
        train_losses = []
        val_losses = []    
        best_val_loss = float('inf')
        patience = self.config.run.patience or 3
        wait = 0
        
        try:

            if (
                not self.config.run.evaluate
                and self.config.run.resume_ckpt_path is not None
            ):
                self.logger.info(
                    f"Loading the checkpoint from path: {self.config.run.resume_ckpt_path}"
                )
                self.load_checkpoint(self.config.run.resume_ckpt_path)

            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if not self._dataloaders.get("train") and not self.config.run.evaluate:
                raise ValueError("Training dataloader is empty")

            # if not self._dataloaders.get("val"):
            #     raise ValueError("Validation dataloader is empty")

            self.logger.info("Start running the training loop")
            self.logger.info(
                f"Start epoch: {self.start_epoch}. Max epoch: {self.max_epoch}"
            )

            self._scaler = torch.amp.GradScaler()
            for epoch in range(self.start_epoch, self.max_epoch):

                # training step
                if not self.config.run.evaluate:
                    self.logger.info(f"Training epoch: {epoch}")
                    train_loss = self.train(epoch)
                    train_losses.append(train_loss)                   
                else:                                                    
                    # evaluation step
                    self.logger.info(f"Evaluation epoch: {epoch}")
                    val_loss = self.eval(epoch)
                    val_losses.append(val_loss)

                    
                    # self.logger.info(f"Train loss: {train_loss}. Eval loss: {val_loss}")
                    
                    if val_loss < best_val_loss:                        
                        best_val_loss = val_loss                    
                        wait = 0
                        self.save_checkpoint(self.model, self.optimizer, 
                                            epoch, val_loss)
                    else:
                        wait += 1
                    
                    if wait >= patience:
                        self.logger.info(f"Early Stopping at epoch: {epoch}")
                        break
                
                # losses = {"Train loss": train_losses, "Val loss": val_losses}
                # plot_losses(losses)
            
            
            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            self.logger.info(f"Finished the training loop in {minutes}:{seconds}")

        except Exception as e:
            self.logger.error(f"Error on runing the agent. Details: {e}")

    def train(self, epoch):

        train_loader = self._dataloaders["train"]

        if len(train_loader) == 0:
            return float("inf")

        self.model.train()
        running_loss = 0.0
        curr_step = 0
        accumulated_gradients = self.config.run.accumulated_gradients or 1
        
        for batch_sample in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            
            batch_sample = prepare_sample(
                batch_sample, cuda_enabled=torch.cuda.is_available()
            )

            self.lr_scheduler.step(cur_epoch=epoch, cur_step=curr_step)

            with torch.amp.autocast("cuda", enabled=self.config.run.amp):
                outputs = self.model(batch_sample)
                loss = outputs["loss"]

            if not torch.isnan(loss).any():                
                if self.config.run.amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimizer)
                else:
                    loss.backward()

                # prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if (curr_step + 1) % accumulated_gradients == 0:
                    if self.config.run.amp:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                curr_step += 1
                running_loss += loss.item()
            else:
                self.logger.info("NaN detected")
                        
        return running_loss / len(train_loader)

    @torch.no_grad()
    def eval(self, epoch):
        running_eval_loss = 0
        val_loader = self._dataloaders["val"]

        if len(val_loader) == 0:
            return float("inf")

        self.model.eval()

        for batch_sample in tqdm(val_loader, desc=f"Evaluating epoch {epoch}"):

            batch_sample = prepare_sample(
                batch_sample, cuda_enabled=torch.cuda.is_available()
            )
            
            with torch.amp.autocast("cuda", enabled=self.config.run.amp):
                outputs = self.model(batch_sample)               
                loss = outputs["loss"]

            if not torch.isnan(loss).any():                
                running_eval_loss += loss.item()
            else:
                self.logger.info("NaN detected")

        return running_eval_loss / len(val_loader)

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
        # raise NotImplementedError

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

                loader = DataLoader(
                    split,
                    batch_size=self.config.datasets[dataset_name].batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=True if is_train else False,
                    collate_fn=collate_fn,
                )
                dataloaders[split.split_name] = loader

        return dataloaders

    def create_optimizer(self):
        self.logger.info("Creating the optimizer")
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
            weight_decay=float(self.config.run.weight_decay),
            betas=(beta1, beta2),
        )

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        return model
    
    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        
        file_name = f'checkpoint_certifiedpt.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

        path = self.config.run.output_dir
        file_and_path = os.path.join(path, file_name)
        torch.save(checkpoint, file_and_path)
        self.logger.info(f"Checkpoint saved at path: {file_and_path}")
    
        
