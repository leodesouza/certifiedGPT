# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
import os
import random
from pathlib import Path
from datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from common.registry import registry

from transformers import AutoTokenizer


class VQAv2Dataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        questions_paths,
        vis_paths,
        annotation_paths,
        split="train",
    ):
        super().__init__(
            vis_processor=vis_processor,
            text_processor=text_processor,
            questions_paths=questions_paths,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths,
        )
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.instruction_template = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}",
        ]

        self.split = split
        questions_dict = {q["question_id"]: q for q in self.questions}

        self.logger.info(
            f"Filter annotations that contains images int the path: {vis_paths}"
        )
        exist_annotation = []

        try:

            for annotation in self.annotations:
                image_id = annotation["image_id"]
                file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
                image_path = os.path.join(self.vis_paths, file_name)
                if os.path.exists(image_path):
                    exist_annotation.append(annotation)
            self.annotations = exist_annotation

            config = registry.get_configuration_class("configuration")
            seed = config.run.seed
            random.seed(seed)

            sample_size = config.datasets.vqav2.sample_size
            if sample_size is not None or sample_size != 0:
                self.logger.info(
                    "Filter annotations based on sample_size hyperparemeter "
                )
                self.logger.info(f"sample_size={sample_size}")
                self.annotations = random.sample(self.annotations, sample_size)

            self.questions = [
                questions_dict[ann["question_id"]]
                for ann in self.annotations
                if ann["question_id"] in questions_dict
            ]
        except Exception as e:
            self.logger.error(f"error on loading the dataset. Details: {e}")

    def get_data(self, index):

        try:
            annotation = self.annotations[index]

            if (
                "image_id" not in annotation
                or "question_id" not in annotation
                or "answers" not in annotation
            ):
                raise ValueError(f"Invalid annotation at index {index}: {annotation}")

            image_id = annotation["image_id"]
            file_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"
            image_file_path = os.path.join(self.vis_paths, file_name)
            image = Image.open(image_file_path).convert("RGB")
            image = self.vis_processor(image)

            question_id = annotation["question_id"]
            question = next(
                filter(lambda q: q["question_id"] == question_id, self.questions), None
            )
            question = self.text_processor(question["question"])

            if question is None or "question" not in question:
                raise ValueError(
                    f"Invalid or missing question for question_id {question_id}"
                )

            all_answers = annotation["answers"]
            num_answer = len(all_answers)

            if num_answer == 0:
                raise ValueError(f"No answers found for question_id {question_id}")

            weight = 1 / num_answer
            answer_weights = {}

            for answer in all_answers:
                answer = answer.get("answer")
                if answer in answer_weights:
                    answer_weights[answer] += weight
                else:
                    answer_weights[answer] = weight

            if not answer_weights:
                raise ValueError(
                    f"No valid answers processed for question_id {question_id}"
                )

            answers = list(answer_weights.keys())
            weights = list(answer_weights.values())
            answer = random.choices(answers, weights=weights, k=1)
            answer = answer[0]
            answer = self.text_processor(answer)

            return {
                "image": image,
                "question": question,
                "question_id": question_id,
                "answer": answer,
            }
        except Exception as e:
            print(f"Error at index:{e}")

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_template).format(data["question"])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data["image"],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": data["answer"],
        }

    @property
    def split_name(self):
        return self.split

    def tokenize(self, x):
        return self._tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )["input_ids"].squeeze(0)


class VQAv2EvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_paths, annotation_paths):
        super().__init__(
            vis_processor=vis_processor,
            text_processor=text_processor,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths,
        )

    def __getitem__(self, index):
        return NotImplementedError
