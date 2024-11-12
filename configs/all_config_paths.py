DATASET_CONFIG_DICT = {"vqav2": "~/projects/certifiedGPT/configs/datasets/vqav2/defaults.yaml"}


def get_database_config_path(key):
    return DATASET_CONFIG_DICT[key]
