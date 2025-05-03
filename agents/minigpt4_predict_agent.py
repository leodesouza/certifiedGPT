# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
from time import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Torch
import torch


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# Pytorch XLA
# from torch_xla import runtime as xr
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.parallel_loader as pl

from agents.base import BaseAgent
from common.metrics import TPUMetrics
from common.registry import registry
from randomized_smoothing.smoothing import Smooth
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import pickle


@registry.register_agent("image_text_eval")
class MiniGPT4PredictionAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_step = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self.build_model()
        self._tpu_metrics = TPUMetrics()
        self.questions_paths = None
        self.annotations_paths = None
        self.smoothed_decoder = Smooth(self._model, self.config.run.noise_level)                
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.results = []
            

    def run(self):
        try:

            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if self.is_main_process():
                if self.config.run.noise_level > 0:
                    print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    print("No noise will be applied to the image inputs")

            self.load_finetuned_model(self._model)
            self.predict()

        except Exception as e:            
            msg = f"Error on agent run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Details: {e}"
            print(msg)
            self.logger.error(msg)

    @torch.no_grad()
    def predict(self):        
        val_loader = self._dataloaders["val"]
                
        n = self.config.run.number_monte_carlo_samples_for_estimation

        if len(val_loader) == 0:
            return float("inf")
        
        self.logger.info(f"Prediction started: {(self.formated_datetime())}")                   

        saved_step = 0
        state = self.load_prediction_state()
        if state is not None:            
            saved_step = state.get("step", 0)                     
            self.results = state.get("prediction_results", [])            
            saved_step += 1 
            print(f"Prediction will be resumed from step: {saved_step}")

        before_time = time()        

        self.model.eval()
        for step, batch_sample in enumerate(val_loader):

            # certify every skip examples and break when step == max
            if step % self.config.run.skip != 0:
                continue

            if step < saved_step:
                continue            
                                                              
            image_id = batch_sample["image_id"]            
            question_id = batch_sample["question_id"]
            question = batch_sample["instruction_input"]
            answers = batch_sample["answer"]                                             
                        
            # eval prediction of smoothed decoder around images
            self.logger.info(f"Prediction Step {step} started") 
            before_time = time()            
            prediction = self.smoothed_decoder.predict(
                batch_sample, n, self.config.run.alpha, batch_size=self.config.run.batch_size
            )

            self.logger.info(f"prediction -- {prediction}")
            self.logger.info(f"answers -- {answers}")

            after_time = time()                        
            time_elapsed = str(timedelta(seconds=(after_time - before_time)))            
            
            correct = False
            if prediction != self.smoothed_decoder.ABSTAIN:                                                                    
                for a in answers: 
                    text = a[0]    
                    self.logger.info(f"text to compare: {text}")                                    
                    similarity_threshold = self.config.run.similarity_threshold            
                    embp = self.sentence_transformer.encode(prediction)
                    embt = self.sentence_transformer.encode(text)                                        
                    print(f"embp shape: {embp.shape}, embt shape: {embt.shape}")
                    similarity = util.cos_sim(embp, embt)
                    similarity_score = similarity.item()                    
                    correct  = similarity_score >= similarity_threshold
                    if correct:
                        break
            
            self.logger.info(f"writing results..")
            self.results.append(f"{step}\t{image_id.item()}\t{question_id.item()}\t{question[0]}\t{answers}\t{prediction}\t{correct}\t{time_elapsed}")                
            self.logger.info(f"Prediction Step {step} ended in {time_elapsed}")
            self.save_prediction_state(step, self.results)

        if self.is_main_process():
            file_path = os.path.join(self.config.run.output_dir,"predict_output.txt")
            file_exists = os.path.exists(file_path)

            with open(file_path, 'a') as f:
                if not file_exists:
                    f.write("step\timageid\tquestion_id\tquestion\tanswer\tpredicted\tcorrect\ttime\n")
                f.write("\n".join(self.results) + "\n")            

        print(f"Prediction ended: {(self.formated_datetime())}")
        self.logger.info(f"Prediction ended: {(self.formated_datetime())}")                   
    
    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    @staticmethod
    def add_noise(batch_sample, noise_level):
        image = batch_sample["image"]
        noise = torch.rand_like(image) * noise_level
        batch_sample["image"] = image + noise

    def _build_datasets(self):
        datasets = dict()
        datasets_config = self.config.datasets
        for name in datasets_config:
            builder = registry.get_builder_class(name)
            builder_instance = builder()
            dataset = builder_instance.build_datasets()
            datasets[name] = dataset

        return datasets

    def create_dataloaders(self, batch_size=-1):
        self.logger.info("building datasets")
        datasets = self._build_datasets()
        dataset_names = sorted(datasets.keys())
        dataloaders = dict()

        for dataset_name in dataset_names:

            dataset = datasets[dataset_name]

            for split in dataset.values():
                num_records = len(split)
                if num_records >= 0:
                    self.logger.info(
                        "Loaded {} records for split {}".format(num_records, split.split_name)
                    )

                self.questions_paths = split.questions_paths
                self.annotations_paths = split.annotations_paths

                is_train = (
                    True if split.split_name in self.config.run.train_splits else False
                )

                collate_fn = getattr(split, "collater", None)
              

                if self.config.run.distributed and dist.is_initialized() and dist.get_world_size() > 1:
                    sampler = DistributedSampler(
                        split,
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=is_train
                    )
                else:
                    sampler = None
                

                loader = DataLoader(
                    split,
                    batch_size=batch_size if batch_size > 0 else self.config.datasets[dataset_name].batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=(True if is_train and not self.config.run.distributed else False),
                    sampler=sampler,
                    collate_fn=collate_fn,
                    drop_last=True
                )
                dataloaders[split.split_name] = loader

        return dataloaders

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        model.to(self.device)
        return model
    
    def save_prediction_state(self, step, certification_results):        
        print("saving state..")   
    
        state = dict()                
        state["step"] = step
        state["prediction_results"] = certification_results                

        rank = dist.get_rank() if dist.is_initialized() else 0
        file_path = os.path.join(self.config.run.output_dir,f"prediction_output_r{rank}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print("state saved!")   
        

    def load_prediction_state(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        file_path = os.path.join(self.config.run.output_dir, f"prediction_output_r{rank}.pkl")

        if not os.path.exists(file_path):
            print(f'file not found: {file_path}')
            return None
        
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        print("prediction_state_loaded")           
        return state       