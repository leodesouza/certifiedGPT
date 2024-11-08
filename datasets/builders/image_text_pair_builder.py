class BaseDataSetBuilder:
    train_dataset_cls = eval_datasets_cls = None, None

    def __init__(self):
        pass


class VQAv2Builder(BaseDataSetBuilder):
    train_dataset_cls =  VQAv2Dataset

    def __init__(self):
        pass
