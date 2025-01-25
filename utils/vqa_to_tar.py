import os 
import json 
from pathlib import Path
import tarfile
from PIL import Image 
from io import BytesIO
from omegaconf import OmegaConf
from common.registry import registry

from utils.generate_subset_vqa import load_dataset_config, open_json_file


def preprocess_vqa_to_tar(output_tar, annotations, questions, images_dir, split):
    with tarfile.open(output_tar, "w") as tar:
        for annotation in annotations:
            image_id = annotation["image_id"]
            question_id = annotation["question_id"]

            file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(images_dir, file_name)
            
            if not os.path.exists(file_name):
                continue

            with Image.open(image_path).convert("RGB") as img:
                img_buffer = BytesIO()
                img.save(img_buffer)
                img_data = img_buffer.getvalue()
            
            question = next((q["question"] for q in questions if q["question_id"] == question_id), None)

            if not question: 
                continue

            answers = annotation["answers"]

            metadata = {
                "question_id": question_id,
                "question": question,
                "answers" : answers,
                "split": split
            }

            tarinfo_img = tarfile.TarInfo(name=f"{question_id}.jpg")
            tarinfo_img.size = len(img_data)
            tar.addfile(tarinfo_img, BytesIO(img_data))

            metadata_str = json.dumps(metadata).encode("utf-8")
            tarinfo_meta = tarfile.TarInfo(name=f"{question_id}.json")
            tarinfo_meta.size = len(metadata_str)
            tar.addfile(tarinfo_meta, BytesIO(metadata_str))

if __name__ == "__main__":
    
    split = "train"

    root_path = Path(__file__).resolve().parent.parent
    root_path = str(root_path)
    config_file_path = os.path.join(root_path, "configs/datasets/vqav2/defaults_vqa.yaml")
    config = load_dataset_config(config_file_path)
    build_info = config.build_info

    
    annotation_path = build_info.annotations[split].path[0]
    file_path = Path(annotation_path)
    folder_path = file_path.parent
    json_file = open_json_file(annotation_path)
    annotations = json_file["annotations"]

    images_path = build_info.images[split].path[0]
    images_dir = os.environ["DATA_DIR"]
    
    
    questions = []
    images_dir = ""
    output_tar = ""

    preprocess_vqa_to_tar(output_tar, annotations, questions, images_dir, split)



