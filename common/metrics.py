import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from common.registry import registry

class TPUMetrics:
    def log_tpu_metrics(self):        
            log = f"Number of compilations: {met.metric_data('CompileTime')[:1]}"
            xm.master_print(log)        