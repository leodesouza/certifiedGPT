# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import os
import datetime
from time import time
from pathlib import Path
import numpy as np
import pickle

# Torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Projeto
from agents.base import BaseAgent
from common.metrics import TPUMetrics  # se não usar, pode remover
from common.registry import registry
from randomized_smoothing.smoothing_v2 import SmoothV2
from bert_score import score
from sentence_transformers import SentenceTransformer, util


@registry.register_agent("image_text_eval")
class MiniGPT4CertifyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self._model = self.build_model()        
        self._tpu_metrics = TPUMetrics()
        self.questions_paths = None
        self.annotations_paths = None        
        self.smoothed_decoder = SmoothV2(self.model, self.config.run.noise_level)        
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(self.device))
        self.results = []        

    def run(self):
        try:
            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if self.is_main_process():
                if self.config.run.noise_level > 0:
                    print(f"[Master] Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    print("[Master] No noise will be applied to the image inputs")

            self.load_finetuned_model(self.model)
            self.certify()

        except Exception as e:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"Error on agent run: {now}. Details: {e}"
            if self.is_main_process():
                print(msg)
            self.logger.error(msg)

    @torch.no_grad()
    def certify(self):
        val_loader = self._dataloaders["val"]

        n0 = self.config.run.number_monte_carlo_samples_for_selection
        n = self.config.run.number_monte_carlo_samples_for_estimation

        if len(val_loader) == 0:
            return float("inf")

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.is_main_process():
            print(f"Certification started: {now}")
        self.logger.info(f"Certification started: {now}")
        
        saved_step = 0        
        state = self.load_certification_state()
        if state is not None:
            saved_step = state.get("step", 0)
            self.results = state.get("certification_results", [])
            saved_step += 1
            if self.is_main_process():
                print(f"Certification will be resumed from step: {saved_step}")

        self.model.eval()

        for step, batch_sample in enumerate(val_loader):

            # certificar a cada 'skip' exemplos
            if step % self.config.run.skip != 0:
                continue

            if step < saved_step:
                continue

            image_id = batch_sample["image_id"]
            question_id = batch_sample["question_id"]
            question = batch_sample["instruction_input"]
            answers = batch_sample["answer"]

            # Certificação
            self.logger.info(f"Certify Step {step} started")
            before_time = time()

            prediction, radius, top1_is_unk = self.smoothed_decoder.certify(
                batch_sample, n0, n, self.config.run.alpha, batch_size=self.config.run.batch_size
            )

            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            correct = False
            if prediction != self.smoothed_decoder.ABSTAIN:
                for a in answers:
                    text = a[0]
                    text = self.smoothed_decoder._normalize_vqa(text)
                    similarity_threshold = self.config.run.similarity_threshold
                    
                    embp = self.sentence_transformer.encode(prediction, convert_to_tensor=True, device=self.device)
                    embt = self.sentence_transformer.encode(text, convert_to_tensor=True, device=self.device)
                    similarity = util.cos_sim(embp, embt)
                    similarity_score = similarity.item()
                    correct = similarity_score >= similarity_threshold
                    if correct:
                        break
            abstain = prediction == self.smoothed_decoder.ABSTAIN

            # Alguns campos podem ser tensores: garantir .item() se necessário
            img_id_val = image_id.item() if torch.is_tensor(image_id) and image_id.numel() == 1 else image_id
            q_id_val = question_id.item() if torch.is_tensor(question_id) and question_id.numel() == 1 else question_id
            q_text = question[0] if isinstance(question, (list, tuple)) else str(question)

            self.results.append(
                f"{step}\t{img_id_val}\t{q_id_val}\t{q_text}\t{answers}\t{prediction}\t{radius:.3}\t{correct}\t{abstain}\t{time_elapsed}\t{top1_is_unk}"
            )
            self.logger.info(f"Certify Step {step} ended in {time_elapsed}")
            self.save_certification_state(step, self.results)

        if self.is_main_process():
            file_path = os.path.join(self.config.run.output_dir, "certify_output.txt")
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a', encoding="utf-8") as f:
                if not file_exists:
                    f.write("step\timageid\tquestion_id\tquestion\tanswer\tlabel\tradius\tcorrect\tabstain\ttime\ttop1_is_unk\n")
                f.write("\n".join(self.results) + "\n")

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.is_main_process():
            print(f"Certification ended: {now}")
        self.logger.info(f"Certification ended: {now}")

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
                    self.logger.info(f"Loaded {num_records} records for split {split.split_name}")

                self.questions_paths = getattr(split, "questions_paths", None)
                self.annotations_paths = getattr(split, "annotations_paths", None)

                is_train = True if split.split_name in self.config.run.train_splits else False
                collate_fn = getattr(split, "collater", None)

                use_distributed = bool(self.config.run.distributed) and dist.world_size() > 1
                sampler = DistributedSampler(
                    split,
                    num_replicas=dist.world_size(),
                    rank=dist.get_rank(),
                    shuffle=True if is_train else False,
                    drop_last=True
                ) if use_distributed else None

                loader = DataLoader(
                    split,
                    batch_size=(batch_size if batch_size > 0 else self.config.datasets[dataset_name].batch_size),
                    num_workers=self.config.run.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    shuffle=(True if is_train and not use_distributed else False),
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

        # Se distribuído, usar DDP
        if bool(self.config.run.distributed) and _get_world_size() > 1:
            # Encontrar device_id local para o processo
            local_rank = int(os.environ.get("LOCAL_RANK", "0")) if torch.cuda.is_available() else None
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if local_rank is not None else None,
                output_device=local_rank if local_rank is not None else None,
                find_unused_parameters=False
            )
        return model

    def save_certification_state(self, step, certification_results):
        
        print("saving state..")

        state = dict()
        state["step"] = step
        state["certification_results"] = certification_results

        rank = dist.get_rank() if dist.is_initialized() else 0
        file_path = os.path.join(self.config.run.output_dir, f"certification_output_r{rank}.pkl")
        os.makedirs(self.config.run.output_dir, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        
        print("state saved!")
        

    def load_certification_state(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        file_path = os.path.join(self.config.run.output_dir, f"certification_output_r{rank}.pkl")

        if not os.path.exists(file_path):            
            print(f'file not found: {file_path}')
            return None

        with open(file_path, 'rb') as f:
            state = pickle.load(f)        
        print("certification_state_loaded")        
        return state
