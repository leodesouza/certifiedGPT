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
from graphs.losses.cross_entropy_loss import CrossEntropyLoss
import torch_xla.test.test_utils as test_utils

# rank and world size are inferred from XLA Device
# source: https://github.com/pytorch/xla/
dist.init_process_group(backend='xla', init_method='xla://')

@registry.register_agent("image_text_finetune")
class MiniGPT4FineTuneAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_epoch = 0
        self.start_step = 0
        self.max_epoch = self.config.run.max_epoch
        self._device = xm.xla_device()    
        self._model = self.build_model()
        self._setup_wandb(self._model)
        self._start_epoch = 0        
        self._tpu_metrics = TPUMetrics()     
                                                              
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

            if self.config.run.debug_graph_computation:
                self.debug_graph_computation()
                return            

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
                    xm.master_print("No noise will be applied to the image inputs")
            
            start_epoch = self.load_checkpoint(self._model, self.optimizer)
            if start_epoch > 0:
                 self.start_epoch = start_epoch + 1                            
        
            xm.master_print(f"Train/Eval started started: {(test_utils.now())}")                   
            xm.master_print(f"Start_epoch: {self.start_epoch}")            
            
            for epoch in range(self.start_epoch, self.max_epoch):                                                
                # training step
                if not self.config.evaluate_only:                    
                    xm.master_print(f"Training epoch: {epoch} started: {test_utils.now()}")
                    epoch_train_loss = self.train(epoch)
                    xm.mark_step()
                    xm.master_print(f"Training epoch: {epoch} ended: {test_utils.now()}")                                        

                if self.config.run.has_val_split:
                                            
                    xm.master_print(f"Evaluation epoch: {epoch} started: {test_utils.now()}")
                    epoch_val_loss = self.eval(epoch)                    
                    xm.mark_step()                    
                    xm.master_print(f"Evaluation epoch: {epoch} ended: {test_utils.now()}")

                    # xm.master_print("Call LR Scheduler Plateau")
                    # self.lr_scheduler_plateau.step(epoch_val_loss)
                                                                            
                    if epoch_val_loss < best_val_loss:                        
                        best_val_loss = epoch_val_loss                    
                        wait = 0
                        # self.save_checkpoint(self.model, epoch)                        
                        self.save_checkpoint_with_optim(self.model, self.optimizer, epoch)                                        
                    else:
                        wait += 1
                    
                    if wait >= patience:
                        self.logger.info(f"Early Stopping at epoch: {epoch}")
                        break

            
                if xm.is_master_ordinal():                                                                           
                    xm.master_print(f"""epoch: {epoch}   
                                     train_loss: {epoch_train_loss}   
                                     val_loss: {epoch_val_loss}""")
                                        
                    # self.save_history(epoch_train_loss, epoch_val_loss)                                                            

                    if self.config.run.wandb:
                                            
                        xm.master_print("Logging the metrics to wandb")
                        wandb.log({
                            "epoch": epoch,
                            "train_loss": epoch_train_loss,
                            "val_loss": epoch_val_loss,
                            "learning_rate":self.optimizer.param_groups[0]["lr"]
                        })                                                

            self.save_checkpoint(self.model, epoch)                                    
            xm.master_print(f"Finished the training loop {test_utils.now()}")                                                
            

        except Exception as e:
              xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")                            

    def maybe_add_noise(self, batch_sample, noise_level):
        
        if noise_level > 0:                  
            image = batch_sample["image"]    
            noise = torch.rand_like(image) * noise_level
            batch_sample["image"] = image + noise
                
    def train(self, epoch):                
        
        train_loader = self._dataloaders["train"]                
        train_loader = pl.MpDeviceLoader(train_loader, self.device)                

        if len(train_loader) == 0:
            return float("inf")                                
        
        running_loss = torch.tensor(0.0, device=self.device)
        total_batches = torch.tensor(0, device=self.device)
        
        accumulated_gradients = self.config.run.accumulated_gradients or 1
        lr = 0.0
               
        self.model.train()

        xm.master_print(f"Train Epoch {epoch} started: {(test_utils.now())}")            
        for step, batch_sample in enumerate(train_loader):
            
            self.maybe_add_noise(batch_sample, self.config.run.noise_level)    

            xm.master_print(f"Processing epoch: {epoch}. step: {step} - {(test_utils.now())}")                       
            
            self.optimizer.zero_grad()                                    
            with xla_amp.autocast(enabled=self.config.run.amp, device=self.device):                                                     
                outputs = self.model(batch_sample)                                                            
            loss = outputs["loss"]                        
            loss.backward()                                 

            if step % accumulated_gradients == 0:                
                xm.reduce_gradients(self.optimizer)                                
                xm.optimizer_step(self.optimizer, barrier=False)                      
                lr = self.lr_scheduler.step(cur_epoch=epoch, cur_step=step)                
                
            xm.mark_step()       

            step_loss = loss.detach()                                        
            if xm.is_master_ordinal() and (step + 1) % 5 == 0:                                
                self._tpu_metrics.log_tpu_metrics("Train", epoch, step, step_loss, lr)                
            running_loss += step_loss
            total_batches += 1
                                                    
        global_train_loss = xm.mesh_reduce("running_loss", running_loss.item(), sum)            
        global_total_batches = xm.mesh_reduce("total_batches", total_batches.item(), sum)

        avg_loss = global_train_loss / global_total_batches
        self.loss_history["train_loss"].append(avg_loss)        
        
        xm.master_print(f"Train Epoch {epoch} ended: {(test_utils.now())}")
                                                 
        return avg_loss                

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

            if xm.is_master_ordinal() and (step + 1) % 5 == 0:
                self._tpu_metrics.log_tpu_metrics("Eval", epoch, step, step_loss, 0)                                     

            running_eval_loss += step_loss
            total_batches += 1

        global_eval_loss = xm.mesh_reduce("running_eval_loss", running_eval_loss.item(), sum)                    
        global_total_batches = xm.mesh_reduce("total_batches", total_batches.item(), sum)
        eval_avg_loss = global_eval_loss / global_total_batches

        xm.master_print(f"Eval Epoch {epoch} ended: {(test_utils.now())}")        
        #self.loss_history["val_loss"].append(eval_avg_loss)        
                                
        return eval_avg_loss
    
    def debug_graph_computation(self):                
                
        train_loader = self.dataloader["train"]                
        train_loader = pl.MpDeviceLoader(train_loader, self.device)                
                               
        self.model.train()

        batch_sample = next(iter(train_loader))
        xm.master_print("Start: debug_graph_computation graph")                       
                                                            
        with xla_amp.autocast(enabled=self.config.run.amp, device=self.device):                                                     
            self.model(batch_sample)                                                                        
        xm.mark_step()                          
                                   
        self.optimizer.zero_grad()
        xm.master_print("End: debug_graph_computation graph")                                                             

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
                    batch_size= batch_size if batch_size > 0 else self.config.datasets[dataset_name].batch_size,
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
    
    def save_checkpoint_with_optim(self, model, optimizer, epoch):        

        if xm.is_master_ordinal():            
            
            xm.master_print(f"Saving the checkpoint with optmizer for epoch: {epoch}")    
                                    
            file_name = self.config.run.checkpoint_name_with_optim
            file_name = f"{file_name}.pth"  

            model_cpu = model.cpu()
            optimizer_cpu = optimizer.state_dict()

            xm.master_print(f"Checkpoint name: {file_name}")    
            checkpoint = {
                'epoch': epoch,                
                'model_state_dict': model_cpu.state_dict(),
                'optimizer_state_dict': optimizer_cpu,                
            }

            path = self.config.run.output_dir
            file_and_path = os.path.join(path, file_name)
            
            xm.master_print(f"Saving Checkpoint in the path: {file_and_path}")   

            self._tpu_metrics.log_checkpoint_saving("Saving checkpoint",epoch=epoch)                
            torch.save(checkpoint, file_and_path, _use_new_zipfile_serialization=False)            
            self._tpu_metrics.log_checkpoint_saving("Checkpoint Saved", epoch=epoch)
                                    
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
        model_cpu = model.cpu()
        param_grads = {
                k: v.requires_grad for (k, v) in model.named_parameters()
            }

        state_dict = model_cpu.state_dict()
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
                #                   
        
             
    
        
