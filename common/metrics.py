import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from common.registry import registry

class TPUMetrics:
    def log_tpu_metrics(self):        
        compile_time = met.metric_data('CompileTime')
        if compile_time is not None:
            log = f"Number of compilations: {compile_time[:1]}"
        else:
            log = "CompileTime metric is not available."
        xm.master_print(log)        