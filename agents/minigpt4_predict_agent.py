import os
from time import time
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from time import time
import pickle

from agents.base import BaseAgent
from common.metrics import TPUMetrics  # You might want to rename this to "DeviceMetrics"
from common.registry import registry
from randomized_smoothing.smoothing import Smooth


@registry.register_agent("image_text_eval")
class MiniGPT4PredictionAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self._tpu_metrics = TPUMetrics()
        self.questions_paths = None
        self.annotations_paths = None
        self.smoothed_decoder = Smooth(self.model, self.config.run.noise_level)
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.results = []

    def run(self):
        try:
            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if dist.get_rank() == 0:
                if self.config.run.noise_level > 0:
                    print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    print("No noise will be applied to the image inputs")

            self.load_finetuned_model(self.model)
            self.predict()

        except Exception as e:
            print(f"Error on agent run: {datetime.datetime.now()}. Details: {e}")
            self.logger.error(f"Error on agent run: {datetime.datetime.now()}. Details: {e}")

    @torch.no_grad()
    def predict(self):
        val_loader = self._dataloaders["val"]

        n = self.config.run.number_monte_carlo_samples_for_estimation

        if len(val_loader) == 0:
            return float("inf")

        print(f"Prediction started: {datetime.datetime.now()}")
        self.logger.info(f"Prediction started: {datetime.datetime.now()}")

        saved_step = 0
        state = self.load_prediction_state()
        if state is not None:
            saved_step = state.get("step", 0)
            self.results = state.get("prediction_results", [])
            saved_step += 1
            print(f"Prediction will be resumed from step: {saved_step}")

        self.model.eval()
        for step, batch_sample in enumerate(val_loader):
            if step % self.config.run.skip != 0 or step < saved_step:
                continue

            image_id = batch_sample["image_id"]
            question_id = batch_sample["question_id"]
            question = batch_sample["instruction_input"]
            answers = batch_sample["answer"]

            self.logger.info(f"Prediction Step {step} started")
            before_time = time()
            prediction = self.smoothed_decoder.predict(
                batch_sample, n, self.config.run.alpha, batch_size=self.config.run.batch_size
            )
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            correct = False
            if prediction != self.smoothed_decoder.ABSTAIN:
                for a in answers:
                    text = a[0]
                    similarity_threshold = self.config.run.similarity_threshold
                    embp = self.sentence_transformer.encode(prediction)
                    embt = self.sentence_transformer.encode(text)
                    similarity = util.cos_sim(embp, embt)
                    similarity_score = similarity.item()
                    correct = similarity_score >= similarity_threshold
                    if correct:
                        break

            self.results.append(f"{step}\t{image_id.item()}\t{question_id.item()}\t{question[0]}\t{answers}\t{prediction}\t{correct}\t{time_elapsed}")
            self.logger.info(f"Prediction Step {step} ended in {time_elapsed}")
            self.save_prediction_state(step, self.results)

        if dist.get_rank() == 0:
            file_path = os.path.join(self.config.run.output_dir, "predict_output.txt")
            file_exists = os.path.exists(file_path)

            with open(file_path, 'a') as f:
                if not file_exists:
                    f.write("step\timageid\tquestion_id\tquestion\tanswer\tpredicted\tcorrect\ttime\n")
                f.write("\n".join(self.results) + "\n")

        print(f"Prediction ended: {datetime.datetime.now()}")
        self.logger.info(f"Prediction ended: {datetime.datetime.now()}")

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        model.to(self.device)
        return model

    def create_dataloaders(self, batch_size=-1):
        self.logger.info("building datasets")
        datasets = self._build_datasets()
        dataset_names = sorted(datasets.keys())
        dataloaders = dict()

        for dataset_name in dataset_names:
            dataset = datasets[dataset_name]

            for split in dataset.values():
                self.questions_paths = split.questions_paths
                self.annotations_paths = split.annotations_paths

                is_train = split.split_name in self.config.run.train_splits

                collate_fn = getattr(split, "collater", None)

                sampler = DistributedSampler(
                    split,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=is_train
                ) if self.config.run.distributed and dist.is_initialized() else None

                loader = DataLoader(
                    split,
                    batch_size=batch_size if batch_size > 0 else self.config.datasets[dataset_name].batch_size,
                    num_workers=self.config.run.num_workers,
                    pin_memory=True,
                    shuffle=(is_train and sampler is None),
                    sampler=sampler,
                    collate_fn=collate_fn,
                    drop_last=True
                )
                dataloaders[split.split_name] = loader

        return dataloaders

    def save_prediction_state(self, step, certification_results):
        state = {
            "step": step,
            "prediction_results": certification_results
        }

        rank = dist.get_rank()
        file_path = os.path.join(self.config.run.output_dir, f"prediction_output_r{rank}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print("Prediction state saved!")

    def load_prediction_state(self):
        rank = dist.get_rank()
        file_path = os.path.join(self.config.run.output_dir, f"prediction_output_r{rank}.pkl")

        if not os.path.exists(file_path):
            print(f'file not found: {file_path}')
            return None

        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        print("Prediction state loaded")
        return state
