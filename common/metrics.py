import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from common.registry import registry
from datetime import datetime
from common.registry import registry
import os
import wandb

from utils.gcsfuse import mount_gcsfuse

class TPUMetrics:

    def __init__(self):
         self.config = registry.get_configuration_class("configuration")
    
    def log_tpu_metrics(self, split, epoch, step, loss, lr):  
   
       timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

       compile_time = met.metric_data('CompileTime')
       if compile_time is not None:
           log_compile_time = f'Number of Compilations: {compile_time[:1]}'
       else:
           log_compile_time = "Compile metric is not available"
              

    #    log_message = "\n".join([
    #         f"Split: {split}",
    #         f"Epoch: {epoch}",
    #         f"Step: {step}",
    #         f"Loss: {loss}",
    #         f"Lr: {lr}",
    #         f"TimeStamp: {timestamp}",
    #         f"{log_compile_time}"           
    #     ])
       
       wandb.log({
            "split": split,
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "timestamp": timestamp,
            "compile_time": compile_time[:1] if compile_time is not None else "N/A"
        })

    #    path = self.config.run.output_dir
    #    if not os.path.exists(path):
    #             mount_gcsfuse()            

    #    file_and_path = os.path.join(path, f'{self.config.run.checkpoint_name}.txt')        
           
    #    if not os.path.exists(file_and_path):
    #       with open(file_and_path, 'w') as f:
    #         pass  
              
    #    with open(file_and_path, 'a') as file:
    #        file.write(log_message + "\n\n")
           
           