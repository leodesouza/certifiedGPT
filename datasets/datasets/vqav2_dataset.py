import webdataset as wds

from datasets.datasets.base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),

        )
