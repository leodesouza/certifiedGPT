# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
import random

from datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from common.registry import registry


class VQAv2Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, questions_paths, vis_paths, annotation_paths, split="train"):
        super().__init__(
            vis_processor=vis_processor, text_processor=text_processor,
            questions_paths=questions_paths,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths
        )

        self.instruction_template = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        self.split = split
        questions_dict = {q["question_id"]: q for q in self.questions}

        exist_questions = []
        exist_annotation = []
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

        self.annotations = random.sample(self.annotations, 200)

        self.questions = [questions_dict[ann['question_id']] for ann in self.annotations
                          if ann['question_id'] in questions_dict]

    def get_data(self, index):
        annotation = self.annotations[index]

        image_id = annotation["image_id"]
        file_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image_file_path = os.path.join(self.vis_paths, file_name)
        image = Image.open(image_file_path).convert("RGB")
        image = self.vis_processor(image)

        question_id = annotation["question_id"]
        question = next(filter(lambda q: q['question_id'] == question_id, self.questions), None)
        question = question['question']
        question = self.text_processor(question)

        all_answers = annotation["answers"]
        num_answer = len(all_answers)
        weight = 1 / num_answer
        answer_weights = {}

        for answer in all_answers:
            answer = answer.get("answer")
            if answer in answer_weights:
                answer_weights[answer] += weight
            else:
                answer_weights[answer] = weight

        answers = list(answer_weights.keys())
        weights = list(answer_weights.values())
        anwser = random.choices(answers, weights=weights, k=1)
        anwser = anwser[0]

        return {
            "image": image,
            "question": question,
            "answer": anwser
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_template).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction": instruction,
            "answer": self.text_processor(data['answer'])
        }


class VQAv2EvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_paths, annotation_paths):
        super().__init__(
            vis_processor=vis_processor, text_processor=text_processor,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths
        )

    def __getitem__(self, index):
        return NotImplementedError
