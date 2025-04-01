# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#
import json
import os
import random
from pathlib import Path
from torch.utils.data import Dataset
from datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from common.registry import registry
import torch_xla.core.xla_model as xm
import collections


class VQAv2Dataset(BaseDataset):
    
    def __init__(
        self,
        vis_processor,
        text_processor,
        questions_paths,
        vis_paths,
        annotation_paths,
        split="train"        
    ):
        super().__init__(
            vis_processor=vis_processor,
            text_processor=text_processor,
            questions_paths=questions_paths,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths,
        )
        
        self.instruction_template = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}",
        ]

        xm.master_print(f'Loading {split} split')
        self.split = split
        questions_dict = {q["question_id"]: q for q in self.questions}

        self.logger.info(
            f"Filter annotations that contains images int the path: {vis_paths}"
        )                
                                    
        try:

            self.questions = []

            xm.master_print(f'Loading annotations...')
            
            for annotation in self.annotations:
                question_id = annotation.get("question_id")
                if question_id is None:
                    self.logger.info(
                        f"Warning: Missing 'question_id' in annotation: {annotation}"
                    )
                    continue

                question = questions_dict.get(question_id)
                if question is None:
                    self.logger.info(
                        f"Warning: Question with 'question_id' {question_id} is missing in questions_dict."
                    )
                    continue

                self.questions.append(question)
                            
            self.logger.info("Loading annotations. Done!")
            xm.master_print(f"Loading {split} annotations. Done!")

            self.questions_dict = {q["question_id"]: q for q in self.questions}            

        except Exception as e:            
            xm.master_print(f"error on loading the dataset. Details: {e}")

    def get_data(self, index):

        try:
            annotation = self.annotations[index]

            if (
                "image_id" not in annotation
                or "question_id" not in annotation
                or "answers" not in annotation
            ):
                raise ValueError(f" Invalid annotation at index {index}: {annotation}")
            
            question_id = annotation["question_id"]
            question = self.questions_dict.get(question_id)
            question = self.text_processor(question["question"])

            if question is None:
                raise ValueError(
                    f"Invalid or missing question for question_id {question_id}"
                )
                        
            image_id = annotation.get("image_id")                                
            file_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(self.vis_paths, file_name)                                            
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)

            all_answers = annotation["answers"]
            num_answer = len(all_answers)

            if num_answer == 0:
                raise ValueError(f"No answers found for question_id {question_id}")
            
            answer_weights = collections.defaultdict(float)

            for answer in all_answers:
                
                answer_confidence = answer.get("answer_confidence")
                answer = answer.get("answer")

                if not answer:
                    continue

                confidence = 0 
                if answer_confidence == 'yes':
                    confidence = 2
                elif  answer_confidence == 'maybe':
                    confidence = 1                

                answer_weights[answer] += confidence
                
            total_weight = sum(answer_weights.values())
            if total_weight > 0:
                for answer in answer_weights:
                    answer_weights[answer] /= total_weight

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
                "image_id": image_id
            }
        except Exception as e:
            print(f"Error at index:{index}{e}")
            return None

    def __getitem__(self, index):
        data = self.get_data(index)        
        instruction = random.choice(self.instruction_template).format(data["question"])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data["image"],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": data["answer"],
            "image_id": data["image_id"]
        }       

    @property
    def split_name(self):
        return self.split


class VQAv2TestDataset(Dataset):
    def __init__(self, questions_path, vis_processor, vis_paths, split):
        self.questions_paths = questions_path
        self.vis_processor = vis_processor
        self.vis_paths = vis_paths
        self.split = split
        self.questions = []
        
        self.logger.info("Loading eval dataset ...")
        self.logger.info("Loading questions json files")
        for question_path in self.questions_paths:
            question = json.load(open(question_path, "r"))
            if isinstance(question, dict):
                self.questions.extend(json.load(open(question_path, "r"))["questions"])

    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        data = self.questions[idx]
        img_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        img_file = f"COCO_{self.split}2015_{img_id:012d}.jpg"
        image_path = os.path.join(self.vis_paths, img_file)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "img_id": img_id,
        }

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger

    @property
    def split_name(self):
        return self.split
    

class VQAv2EvalDataset(BaseDataset):
    
    def __init__(
        self,
        vis_processor,
        text_processor,
        questions_paths,
        vis_paths,
        annotation_paths,
        split="train"        
    ):
        super().__init__(
            vis_processor=vis_processor,
            text_processor=text_processor,
            questions_paths=questions_paths,
            vis_paths=vis_paths,
            annotation_paths=annotation_paths,
        )
        
        self.instruction_template = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}",
        ]

        xm.master_print(f'Loading {split} split')
        self.split = split
        questions_dict = {q["question_id"]: q for q in self.questions}

        self.logger.info(
            f"Filter annotations that contains images int the path: {vis_paths}"
        )                
                                    
        try:

            self.questions = []

            xm.master_print(f'Loading annotations...')
            
            for annotation in self.annotations:
                question_id = annotation.get("question_id")
                if question_id is None:
                    self.logger.info(
                        f"Warning: Missing 'question_id' in annotation: {annotation}"
                    )
                    continue

                question = questions_dict.get(question_id)
                if question is None:
                    self.logger.info(
                        f"Warning: Question with 'question_id' {question_id} is missing in questions_dict."
                    )
                    continue

                self.questions.append(question)
                            
            self.logger.info("Loading annotations. Done!")
            xm.master_print(f"Loading {split} annotations. Done!")

            self.questions_dict = {q["question_id"]: q for q in self.questions}            

        except Exception as e:            
            xm.master_print(f"error on loading the dataset. Details: {e}")

    def get_data(self, index):

        try:
            annotation = self.annotations[index]

            if (
                "image_id" not in annotation
                or "question_id" not in annotation
                or "answers" not in annotation
            ):
                raise ValueError(f" Invalid annotation at index {index}: {annotation}")
            
            question_id = annotation["question_id"]
            question = self.questions_dict.get(question_id)
            question = self.text_processor(question["question"])

            if question is None:
                raise ValueError(
                    f"Invalid or missing question for question_id {question_id}"
                )
                        
            image_id = annotation.get("image_id")                                
            file_name = f"COCO_{self.split}2014_{image_id:012d}.jpg"
            image_path = os.path.join(self.vis_paths, file_name)                                            
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)

            all_answers = annotation["answers"]
            num_answer = len(all_answers)
            answers = []

            if num_answer == 0:
                raise ValueError(f"No answers found for question_id {question_id}")
                        

            confidence_count = 0
            for answer in all_answers:

                if confidence_count == 2:
                    break
                                
                answer_confidence = answer.get("answer_confidence")
                answer = answer.get("answer")

                if not answer:
                    continue
                
                if answer_confidence == 'yes':
                    confidence_count += 1
                    answer = self.text_processor(answer)
                    answers.append(answer)                                                                      
            return {
                "image": image,
                "question": question,
                "question_id": question_id,
                "answer": answers,
                "image_id": image_id
            }
        
        except Exception as e:
            print(f"Error at index:{index}{e}")
            return None

    def __getitem__(self, index):
        data = self.get_data(index)        
        instruction = random.choice(self.instruction_template).format(data["question"])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data["image"],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": data["answer"],
            "image_id": data["image_id"]
        }       

    @property
    def split_name(self):
        return self.split
