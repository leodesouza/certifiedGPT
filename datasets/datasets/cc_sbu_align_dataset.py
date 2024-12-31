import os
from datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import json


class CCSbuDataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        questions_paths,
        vis_paths,
        annotation_paths,
        split="train",
    ):
       
        self.logger.info("Loading dataset ...")
        self.annotations = []

        self.logger.info("Loading annotations json files")
        ann_path = json.load(open(annotation_paths, "r"))
        if isinstance(ann_path, dict):
                self.annotations.extend(json.load(open(ann_path, "r"))["annotations"])
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
    

def __getitem__(self, index):
             
    ann = self.annotation[index]

    img_file = '{}.jpg'.format(ann["image_id"])
    image_path = os.path.join(self.vis_root, img_file)
    image = Image.open(image_path).convert("RGB")

    image = self.vis_processor(image)
    caption = ann["caption"]

    return {
        "image": image,
        "answer": caption,
        "image_id": self.img_ids[ann["image_id"]],
    }
    