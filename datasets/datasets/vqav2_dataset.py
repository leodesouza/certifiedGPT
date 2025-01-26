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
import pickle
import torch_xla.core.xla_model as xm


class VQAv2Dataset(BaseDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        questions_paths,
        vis_paths,
        annotation_paths,
        split="train",
        cache_dir="/home/leonardosouza/cache/certifiedgpt/images"
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
        exist_annotation = []

        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / f"{split}_images.pkl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._images =[]
        self.images =[]

        # if self.cache_file.exists():            
        #     xm.master_print("loading images from cache")
        #     with open(self.cache_file,"rb") as f:
        #         self.images = pickle.load(f)
        
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

                image_id = annotation.get("image_id")
                if image_id is None:
                    print(f"Warning: Missing 'image_id' in annotation: {annotation}")
                    continue
                
                file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
                image_path = os.path.join(self.vis_paths, file_name)
                
                if not self.images:                    
                    image = Image.open(image_path).convert("RGB")
                    image = self.vis_processor(image)
                    self._images.append(
                        {
                            "question_id": question_id,
                            "image": image
                        }
                    )            

            if self._images:
                self.images = self._images
                # with open(self.cache_file, "wb") as f:
                #     pickle.dump(self.images, f)
                # self.logger.info(f"cached images to {self.cache_dir}")

            self.logger.info("Loading annotations. Done!")
            xm.master_print(f"Loading {split} annotations. Done!")

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
            question = next(
                filter(lambda q: q["question_id"] == question_id, self.questions), None
            )
            question = self.text_processor(question["question"])

            if question is None:
                raise ValueError(
                    f"Invalid or missing question for question_id {question_id}"
                )
                        
            image = next(
                filter(lambda q: q["question_id"] == question_id, self.images), 
                None
            )

            if image is None:
                raise ValueError(f"Image was not found for question_id: {question_id}")
            
            image = image["image"]
            
            all_answers = annotation["answers"]
            num_answer = len(all_answers)

            if num_answer == 0:
                raise ValueError(f"No answers found for question_id {question_id}")

            weight = 1 / num_answer
            answer_weights = {}

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

                
                if answer in answer_weights:
                    answer_weights[answer] += weight * confidence
                else:
                    answer_weights[answer] = weight * confidence


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
        }

    @property
    def split_name(self):
        return self.split    


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
