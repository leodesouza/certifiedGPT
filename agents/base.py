# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "Pytorch-Project-Template" project.
# See LICENSE_Template.md for the full license text or visit the repo at:
# https://github.com/moemen95/Pytorch-Project-Template/
#

import shutil
import logging
import torch
import torch_xla.core.xla_model as xm
import common
from common.registry import registry
import torch_xla.debug.profiler as xp
import torch_xla.test.test_utils as test_utils
import os
import json



class BaseAgent:
    def __init__(self):
        self.lr_sched = None
        self.lr_sched_plateau = None
        self._optimizer = None
        self._scaler = None
        self._model = None
        self._device = None
        self.config = registry.get_configuration_class("configuration")
        self._dataloaders = None
        self.loss_history = {            
            "train_loss": [],
            "val_loss": []
        }

    def load_checkpoint(self, model, optimizer, copy=False):          
          output_dir = self.config.run.output_dir
          if not output_dir: 
              raise ValueError("output_dir None") 
          
          resume_ckpt_path = f"{self.config.run.resume_ckpt_path}.pth"          
          file_and_path = os.path.join(output_dir, resume_ckpt_path)  

          local_dir = "/tmp"              
          local_resume_path = os.path.join(local_dir, "finetuning_resume.pth")
          os.makedirs(local_dir, exist_ok=True)                              
          if xm.is_master_ordinal() and os.path.exists(file_and_path):
              if copy:                                    
                if not os.path.exists(local_resume_path):
                    xm.master_print(f"Copying checkpoint from {file_and_path} to {local_resume_path}")
                    shutil.copy(file_and_path, local_resume_path)
                    xm.master_print("Checkpoint copied")
              else:
                local_resume_path = file_and_path

          xm.rendezvous("Loading Checkpoint") # sync all process
          
          if os.path.exists(local_resume_path):
              xm.master_print(f"Loading checkpoint from {local_resume_path}")                            
              
              checkpoint = torch.load(local_resume_path, map_location=torch.device('cpu'))              
              xm.rendezvous("Checkpoint loaded") # sync all process

              xm.master_print("Loading model state")         
              model.load_state_dict(checkpoint['model_state_dict'], strict=False)
              
              xm.master_print("Loading optimizer state")         
              load_state_dict = checkpoint['optimizer_state_dict']
              optimizer.load_state_dict({k: v.cpu() if isinstance(v, torch.Tensor) else v for k,v in load_state_dict.items()})

              start_epoch = checkpoint['epoch'] + 1 

              xm.master_print(f"Resume Training from Start_Epoch:{start_epoch}")
              
              return start_epoch
          else:
              return 0                                                                           

    def save_checkpoint(self, model, optimizer, epoch, loss, file_name="checkpoint.pth.bar", is_best=False):
        """

        :param file_name:  name of the checkpoint to save
        :param is_best:  bool indicating if the checkpoint is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """

    def train(self, epoch):
        """
        train a specific epoch
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
        return self._device

    @property
    def model(self):                
        return self._model

    # @property
    # def scaler(self):
    #     amp = self.config.run.amp
    #     if amp:
    #         self._scaler = torch.amp.GradScaler()
    #     return self.scaler

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    def build_model(self):
        """
        Return the current instance of model to train
        :return: ModelBase
        """
        return NotImplementedError()

    @property
    def optimizer(self):
        if self._optimizer is None:            
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():                                

                if not p.requires_grad:
                    continue  # frozen weights                
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            #self.logger.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run.init_lr),
                weight_decay=float(self.config.run.weight_decay),
                betas=(0.9, beta2),
                foreach=False
            )

        return self._optimizer

    @property
    def lr_scheduler_plateau(self):

        if self.lr_sched_plateau is None:
            self.lr_sched_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1, # reduce LR by 10x
                patience=2,  # epochs that scheduler will wait to check and aplly lr reducing
                threshold=0.0001, # minimum change to qualify as "improvement"
                cooldown=1, # epochs that scheduler will pause lr checkings
                min_lr=self.config.run.min_lr
            )

        return self.lr_sched_plateau

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self.lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run.lr_sched)

            max_epoch = self.config.run.max_epoch
            min_lr = self.config.run.min_lr
            init_lr = self.config.run.init_lr

            # optional parameters
            decay_rate = self.config.run.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run.get("warmup_start_lr", -1)
            warmup_steps = self.config.run.get("warmup_steps", 0)
            warmup_max_lr = self.config.run.get("warmup_max_lr", 0)
            iters_per_epoch = self.config.run.get("iters_per_epoch", None)                                    

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self._dataloaders["train"])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self.lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
                warmup_max_lr=warmup_max_lr
            )

        return self.lr_sched

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger  

    def save_history(self, train_loss, val_loss):        
        self.loss_history["train_loss"].append(train_loss)
        self.loss_history["val_loss"].append(val_loss)
        path = self.config.run.output_dir
        file_name_path = os.path.join(path, "loss_history.json")
        with open(file_name_path, "w") as f:
            json.dump(self.loss_history, f)
    
    def load_history(self):
        path = self.config.run.output_dir
        file_name_path = os.path.join(path, "loss_history.json")
        if os.path.exists(file_name_path):
            with open(file_name_path, "r") as f:
                self.loss_history = json.load(f)                
    
    def plot_result(self):                
        train_loss = self.loss_history["train_loss"]
        val_loss = self.loss_history["val_loss"]

        

#         train_loss = loss_history["train_loss"]
# val_loss = loss_history["val_loss"]

# # Plot the loss curves
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", marker="o")
# plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", marker="s")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training & Validation Loss")
# plt.legend()
# plt.grid()
# plt.show()





        


