# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
import os 
import time
import torch
import torch.distributed as dist
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils

from torch.utils.data import DataLoader, DistributedSampler
import torch_xla.amp as xla_amp

from agents.base import BaseAgent
from common.metrics import TPUMetrics
from common.registry import registry
from graphs.losses.cross_entropy_loss import CrossEntropyLoss
from tqdm import tqdm
import wandb
import datetime

# torch.autograd.set_detect_anomaly(False) doesnt work for TPU

# rank and world size are inferred from XLA Device
# source: https://github.com/pytorch/xla/
dist.init_process_group(backend='xla', init_method='xla://')


def train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device, 
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer()
    )

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


def move_to_xla(sample):
    def _move_to_xla(tensor):
        return tensor.to(xm.xla_device())

    return apply_to_sample(_move_to_xla, sample)


def prepare_sample(samples, xla_enabled=True):
    if xla_enabled:
        samples = move_to_xla(samples)
    return samples



@registry.register_agent("image_text_finetune")
class MiniGPT4FineTuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.max_epoch = self.config.run.max_epoch
        self._device = xm.xla_device()    
        self._model = self.build_model()
        self._setup_wandb(self._model)
        self._start_epoch = 0        
        self._tpu_metrics = TPUMetrics() 
        
        xm.broadcast_master_param(self._model) 

        # if xm.is_master_ordinal():
        #     self.writer = test_utils.get_summary_writer(logdir=self.config.run.output_dir)  

    def run(self):        
        best_val_loss = float('inf')                
        patience = self.config.run.patience or 3
        wait = 0
        step = 1
        epoch_train_loss = 0
        epoch_val_loss = 0 
        
        try:
            
            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if not self._dataloaders.get("train") and not self.config.run.evaluate:
                raise ValueError("Training dataloader is empty")
            

            self.logger.info("Start running the training loop")
            self.logger.info(
                f"Start epoch: {self.start_epoch}. Max epoch: {self.max_epoch}"
            )
            
            if xm.is_master_ordinal():     
                if self.config.run.noise_level > 0:
                    xm.master_print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")                           
                else:                    
                    xm.master_print(f"No noise will be applied to the image inputs")
            
            for epoch in range(self.start_epoch, self.max_epoch):
                
                                
                # training step
                if not self.config.evaluate_only:                    
                    xm.master_print(f"Training epoch: {epoch} started: {test_utils.now()}")
                    epoch_train_loss = self.train(epoch)
                    xm.master_print(f"Training epoch: {epoch} ended: {test_utils.now()}")                    

                if self.config.run.has_val_split:
                                            
                    xm.master_print(f"Evaluation epoch: {epoch} started: {test_utils.now()}")
                    epoch_val_loss = self.eval(epoch)                    
                    xm.master_print(f"Evaluation epoch: {epoch} ended: {test_utils.now()}")
                                                                            
                    if epoch_val_loss < best_val_loss:                        
                        best_val_loss = epoch_val_loss                    
                        wait = 0
                        self.save_checkpoint(self.model, epoch)
                    else:
                        wait += 1
                    
                    if wait >= patience:
                        self.logger.info(f"Early Stopping at epoch: {epoch}")
                        break

            
                if xm.is_master_ordinal():                        

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")         
                    self.logger.info(f"current_time: {current_time}. epoch: {epoch} executed.")   

                                    
                    xm.master_print(f"""epoch: {epoch}   
                                     train_loss: {epoch_train_loss}   
                                     val_loss: {epoch_val_loss}""")
                    
                    if self.config.run.wandb:
                                            
                        xm.master_print(f"Logging the metrics to wandb")
                        wandb.log({
                            "epoch": epoch,
                            "train_loss": epoch_train_loss,
                            "val_loss": epoch_val_loss,
                            "learning_rate":self.optimizer.param_groups[0]["lr"]
                        })

                        self._tpu_metrics.log_tpu_metrics(step)
                        step += 1                       
            
            
            
            xm.master_print(f"Finished the training loop {test_utils.now()}")
            # test_utils.close_summary_writer(self.writer)
            

        except Exception as e:
              xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")

    def add_noise(self, image_inputs, noise_level):
        
        if noise_level < 0:
            raise ValueError("Noise level must be greater than 0")        

        noised_image_inputs = image_inputs + torch.rand_like(image_inputs) * noise_level
        
        return noised_image_inputs
    
    def train(self, epoch):
        
        train_loader = self._dataloaders["train"]
        train_loader = pl.MpDeviceLoader(train_loader, self.device)
        
        if len(train_loader) == 0:
            return float("inf")
        
        # tracker = xm.RateTracker()
        self.model.train()
        running_loss = 0.0
        
        accumulated_gradients = self.config.run.accumulated_gradients or 1
        noise_level = self.config.run.noise_level
                                
        for step, batch_sample in enumerate(train_loader):
            
            if step % accumulated_gradients == 0:
                self.optimizer.zero_grad() 

            if noise_level > 0:                
                batch_sample["image"] = self.add_noise(image_inputs, noise_level)
                                                                                        
            
            with xla_amp.autocast(enabled=self.config.run.amp, device=self.device): 
                outputs = self.model(batch_sample)
                loss = outputs["loss"]                                                                
            
            loss.backward() 

            if (step + 1) % accumulated_gradients == 0:
                xm.optimizer_step(self.optimizer, barrier=False)
                self.lr_scheduler.step(cur_epoch=epoch, cur_step=step)                
                xm.mark_step()
                                  
            # loss.detach() to avoid unnecessary computation graph retention                                    
            running_loss += loss.detach().item()                                                                                          
                                        
        avg_loss = xm.mesh_reduce("running_loss", running_loss, lambda x: sum(x) / len(x)) / len(train_loader)            
        
        xm.master_print(f"current_time: {(test_utils.now())}. Step: {step} executed.")
                                                 
        return avg_loss

    @torch.no_grad()
    def eval(self, epoch):
        running_eval_loss = 0

        val_loader = self._dataloaders["val"]
        val_loader = pl.MpDeviceLoader(val_loader, self.device)        
        noise_level = self.config.run.noise_level

        if len(val_loader) == 0:
            return float("inf")

        self.model.eval()

        for _, batch_sample in enumerate(val_loader):
            
            if noise_level > 0:
                image_inputs = batch_sample["image"]
                noised_image_inputs = self.add_noise(image_inputs, noise_level)
                batch_sample["image"] = noised_image_inputs                        
            
            outputs = self.model(batch_sample)               
            loss = outputs["loss"]                
            running_eval_loss += loss.item()                        

            xm.mark_step()

        eval_avg_loss = xm.mesh_reduce("running_eval_loss", running_eval_loss, lambda x: sum(x) / len(x)) / len(val_loader)                    
                                
        return eval_avg_loss
    

    def validate(self):
        """
        One cycle of model validation
        :return:
        """

    def finalize(self):
        pass
        # if self.writer:
        #     self.writer.close()
    

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


                sampler = DistributedSampler(
                    split,
                    num_replicas=xr.world_size(),
                    rank=xm.runtime.global_ordinal(),
                    shuffle=True if is_train else False
                ) if self.config.run.distributed and xr.world_size() > 1 else None

                
                loader = DataLoader(
                    split,
                    batch_size=self.config.datasets[dataset_name].batch_size,
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
        model.to(self.device)
        return model
    
    def save_checkpoint(self, model, epoch):        

        if xm.is_master_ordinal():

            self.logger.info("save in the main process")    

            model_state_dict = self.return_state_dict_without_grad(model)
                        
            file_name = self.config.run.checkpoint_name
            file_name = f"{file_name}_epoch_{epoch}_noise_{self.config.run.noise_level}.pth"  

            self.logger.info(f"checkpoint name: {file_name}")    
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,                                                                                
            }

            path = self.config.run.output_dir

            file_and_path = os.path.join(path, file_name)
            self.logger.info(f"Saving Checkpoint in the path: {file_and_path}")   
            os.makedirs(path, exist_ok=True)    
            
            torch.save(checkpoint, file_and_path)
            self.logger.info(f"Checkpoint saved at path: {file_and_path}")

        #synchronize all the processes
        xm.rendezvous("checkpoint_saved")

    def return_state_dict_without_grad(self, model):
        """
        Return the state_dict without the parameters that do not require gradients
        """
        param_grads = {
                k: v.requires_grad for (k, v) in model.named_parameters()
            }

        state_dict = self.model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grads.keys() and not param_grads[k]:
                del state_dict[k] 

        return state_dict

  
    def _setup_wandb(self, model):
        if self.config.run.wandb and xm.is_master_ordinal():
            self.logger.info("Start wandb")
            wandb.login(key=self.config.run.wandb_api_key)
            wandb.init(                
                project="certifiedgpt",                
                name=self.config.run.wandb_name,
                job_type="train" if self.config.run.evaluate else "eval",
                config=self.config
            )
            
            
            wandb.watch(model)  

            if(not self.config.run.evaluate):
                # Define metrics once during initialization    
                wandb.define_metric("train_loss", step_metric="epoch")
                wandb.define_metric("val_loss", step_metric="epoch")
                wandb.define_metric("learning_rate", step_metric="epoch")

            # validation metric
            if(self.config.run.evaluate):
                wandb.define_metric("accuracy", step_metric="epoch")
                # wandb.define_metric("perplexity", step_metric="epoch")

             
    
        
