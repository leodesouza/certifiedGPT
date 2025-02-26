import torch_xla.core.xla_model as xm
import subprocess
from common.registry import registry

def mount_gcsfuse(bucket_name="certifiedgpt_storage", mount_point="~/storage"):
    logger = registry.get_configuration_class("logger")
    try:
        subprocess.run(["gcsfuse", bucket_name, mount_point], check=True)
        xm.master_print(f"Successfully mounted {bucket_name} at {mount_point}")        
    except subprocess.CalledProcessError as e:        
        logger.error(f"Error mounting {bucket_name}: {e}", exc_info=True)

# Run the function

if __name__ == "__main__":
    mount_gcsfuse()
