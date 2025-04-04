import json
import logging
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from common.registry import registry
from common import utils
from configs.all_config_paths import get_database_config_path, DATASET_CONFIG_DICT
from sklearn.model_selection import train_test_split
import shutil

OmegaConf.register_new_resolver("env", lambda key: os.environ.get(key, None), replace=True)


def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)


def register_variables():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", root_dir)
    registry.register("MAX_INT", sys.maxsize)
    registry.register("SPLIT_NAMES", ["train", "val", "test"])


def default_config_path(key="vqav2"):
    return utils.get_abs_path(DATASET_CONFIG_DICT[key])


def load_dataset_config(config_path):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    config_datasets = config.datasets
    config_datasets = config_datasets[list(config_datasets.keys())[0]]
    return config_datasets


def open_json_file(path):
    json_file = json.load(open(path, "r"))
    return json_file


def sample(build_info, split, new_file_name):
    sample_path = "/home/leonardosouza/projects/datasets/vqav2/annotations/val/sample_v2_mscoco_val2014_annotations.json"
    sample_file = open_json_file(sample_path)
    images_id_from_sample = [q["image_id"] for q in sample_file["annotations"]]

    exist_annotation = []
    annotation_path = build_info.annotations[split].path[0]
    file_path = Path(annotation_path)
    folder_path = file_path.parent
    json_file = open_json_file(annotation_path)

    images_path = build_info.images[split].path[0]
    data_dir = os.environ["DATA_DIR"]
    new_images_path = os.path.join(data_dir, "images/sample_10k", split)

    for annotation in json_file['annotations']:
        image_id = annotation["image_id"]
        if image_id in images_id_from_sample:
            continue
        file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(images_path, file_name)
        if os.path.exists(image_path):
            exist_annotation.append(annotation)

    annotations = exist_annotation
    questions_type = [ann['question_type'] for ann in annotations]

    splits = []
    remaining_annotations = annotations
    for _ in range(4):
        sp, remaining_annotations = train_test_split(
            remaining_annotations,
            train_size=5000,
            random_state=42,
            shuffle=True,
            stratify=questions_type)
        splits.append(sp)
        questions_type = [ann['question_type'] for ann in remaining_annotations]

    # test_size=0.977 for training split
    # questions_type = [ann['question_type'] for ann in annotations]
    # train_or_val_split, _ = train_test_split(
    #     annotations,
    #     test_size=0.977,
    #     random_state=42,
    #     shuffle=True,
    #     stratify=questions_type)

    # for ann in train_or_val_split:
    #     image_id = ann["image_id"]
    #     file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
    #     image_path = os.path.join(images_path, file_name)
    #     shutil.copy(image_path, new_images_path)

    step = 1
    for sp in splits:
        new_images_path = os.path.join(data_dir, f"images/val_{step}")
        if not os.path.exists(new_images_path):
            os.makedirs(new_images_path)
        for ann in sp:
            image_id = ann["image_id"]
            file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(images_path, file_name)
            shutil.copy(image_path, new_images_path)
        json_file['annotations'] = sp
        file_and_path = os.path.join(folder_path, f"sample_v2_mscoco_val2014_annotations__{step}.json")
        with open(file_and_path, "w") as file:
            json.dump(json_file, file, indent=4)
        step += 1

    # json_file['annotations'] = train_or_val_split
    # # file_and_path = os.path.join(folder_path, new_file_name)
    # # with open(file_and_path, "w") as file:
    # #     json.dump(json_file, file, indent=4)


def sample_test_dataset(build_info, split, new_file_name):
    exist_questions = []
    question_path = build_info.questions[split].path[0]
    file_path = Path(question_path)
    folder_path = file_path.parent
    json_file = open_json_file(question_path)

    images_path = build_info.images[split].path[0]
    data_dir = os.environ["DATA_DIR"]
    new_images_path = os.path.join(data_dir, "images/sample_test_10k/")

    for question in json_file['questions']:
        image_id = question["image_id"]
        file_name = f"COCO_{split}2015_{image_id:012d}.jpg"
        image_path = os.path.join(images_path, file_name)
        if os.path.exists(image_path):
            exist_questions.append(question)

    questions = exist_questions

    # test_size=0.977 for training split
    train_or_val_split, _ = train_test_split(
        questions,
        test_size=0.977,
        random_state=42,
        shuffle=True)

    for question in train_or_val_split:
        image_id = question["image_id"]
        file_name = f"COCO_{split}2015_{image_id:012d}.jpg"
        image_path = os.path.join(images_path, file_name)
        shutil.copy(image_path, new_images_path)

    json_file['questions'] = train_or_val_split
    file_and_path = os.path.join(folder_path, new_file_name)
    with open(file_and_path, "w") as file:
        json.dump(json_file, file, indent=4)


def generate_random_samples():
    setup_logger()
    root_path = Path(__file__).resolve().parent.parent
    root_path = str(root_path)
    config_file_path = os.path.join(root_path, "configs/datasets/vqav2/defaults_vqa.yaml")
    config = load_dataset_config(config_file_path)
    build_info = config.build_info

    logging.info('generate val sample')
    sample(build_info, 'val', "sample_v2_mscoco_train2014_annotations.json")

    # logging.info('generate validation sample')
    # sample_test_dataset(build_info, 'test', "sample_v2_OpenEnded_mscoco_test2015_questions.json")

    logging.info('process finished')


if __name__ == "__main__":
    register_variables()
    generate_random_samples()
