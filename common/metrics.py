import torch_xla.debug.metrics as metrics
import wandb

class TPUMetrics:
    def log_tpu_metrics(self, step):
        metrics_report = metrics.metrics_report
        
        utilization = self.parse_tpu_metrics(metrics_report)

        wandb.log(utilization, step=step)

    def parse_tpu_metrics(self, metrics_report):
        metrics = {}
        for line in metrics_report.splitlines():
            if "utilization" in line:
                key, value = line.split(":")
                metrics[key.strip()] = float(value.strip())
        return metrics 


