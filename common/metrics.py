import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from common.registry import registry
from datetime import datetime
from common.registry import registry
import os

class TPUMetrics:

    def __init__(self):
         self.config = registry.get_configuration_class("configuration")
    
    def log_tpu_metrics(self, epoch, step):  

       xm.master_print(f" --> log_tpu_metrics")      
       timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

       compile_time = met.metric_data('CompileTime')
       if compile_time is not None:
           log_compile_time = f'Number of Compilations: {compile_time[:1]}'
       else:
           log_compile_time = "Compile metric is not available"
        
       device_status = met.metric_data("DeviceStatus")
       if device_status is not None:
           log_device_status = f'Device Status: {device_status}'
       else:
           log_device_status = "DeviceStatus metric is not available"

        
       memory_usage = met.metric_data('MemoryUsage')
       if memory_usage is not None:
            log_memory_usage = f"Memory Usage: {memory_usage} MB"
       else:
            log_memory_usage = "MemoryUsage metric is not available."

        
       tpu_utilization = met.metric_data('TPUUtilization')
       if tpu_utilization is not None:
            log_tpu_utilization = f"TPU Utilization: {tpu_utilization}%"
       else:
            log_tpu_utilization = "TPUUtilization metric is not available."

       log_message = "\n".join([
            f"Epoch{epoch} - Step:{step}",
            f"TimeStamp: {timestamp}",
            f"{log_compile_time}",
            f"{log_device_status}",
            f"{log_memory_usage}",
            f"{log_tpu_utilization}"
        ])
       
       path = self.config.run.output_dir
       file_and_path = os.path.join(path, f'{self.config.run.checkpoint_name}.txt')
       xm.master_print(f"file_and_path {file_and_path}")   
       os.makedirs(path, exist_ok=True)  
    
       if not os.path.exists(file_and_path):
          with open(file_and_path, 'w') as f:
            pass  
       
       xm.master_print(f"abrindo arquivo")  
       with open(file_and_path, 'a') as file:
           file.write(log_message)
           
           