import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

class TPUMetrics:
    def log_tpu_metrics(self):        
            xm.master_print(f"Number of compilations: {met.metric_data('CompileTime')[:1]}")                                                                                                                    
    