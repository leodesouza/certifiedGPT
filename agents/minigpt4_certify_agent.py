# SPDX-License-Identifier: BSD-3-Clause
# Portions derived from MiniGPT-4 (see LICENSE)

import os
import time
from pathlib import Path
from datetime import timedelta
import numpy as np
import pickle
import logging

import transformers
transformers.logging.set_verbosity_error()

# Torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from common.vqa_tools.vqa import VQA
from common.vqa_tools.vqa_eval import VQAEval
from agents.base import BaseAgent
from common.registry import registry

from graphs.models.minigpt4.conversation.conversation import CONV_VISION_LLama2

from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab', quiet=True)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_master():
    return get_rank() == 0


@registry.register_agent("image_text_eval")
class MiniGPT4EvalAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_step = 0

        # device e ranks vindos do registry (setados no launch.py GPU)
        self.device = registry.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.rank = registry.get("rank", 0)
        self.world_size = registry.get("world_size", 1)

        self._model = self.build_model()
        self._questions_paths = None
        self._annotations_paths = None
        self._log = []
        self._smooth_fn = SmoothingFunction().method1
        self._predictions = []
        self._ground_truths = []
        self._question_ids = []

    def run(self):
        try:
            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if is_master():
                if self.config.run.noise_level > 0:
                    self.logger.info(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    self.logger.info("No noise will be applied to the image inputs")

            self.load_finetuned_model(self._model)
            self.eval(self._dataloaders)
        except Exception as e:
            self.logger.exception(f"Error on agent run: {e}")

    @torch.no_grad()
    def eval(self, dataloader):
        val_loader = dataloader["val"]
        if len(val_loader) == 0:
            return float("inf")

        conv_temp = CONV_VISION_LLama2.copy()
        conv_temp.system = ""

        if is_master():
            self.logger.info("Eval started")

        before_time = time.time()
        self.model.eval()

        saved_step = 0
        state = self.load_eval_state()
        if state is not None:
            saved_step = state.get("step", 0)
            self._predictions = state.get("predictions", [])
            self._ground_truths = state.get("ground_truths", [])
            self._question_ids = state.get("question_ids", [])
            saved_step += 1
            if is_master():
                self.logger.info(f"Eval will be resumed from step: {saved_step}")

        for step, batch_sample in enumerate(val_loader):
            # manter o mesmo comportamento original (avaliar de 10 em 10)
            if step % 10 != 0:
                continue
            if step < saved_step:
                continue

            if is_master():
                self.logger.info(f"Eval step: {step}")

            self.maybe_add_noise(batch_sample, self.config.run.noise_level)

            image = batch_sample["image"].to(self.device, non_blocking=True)
            image_ids = batch_sample["image_id"].tolist()
            question_ids = batch_sample["question_id"].tolist()
            questions = batch_sample["instruction_input"]
            answers = batch_sample["answer"]

            texts = self.prepare_texts(questions, conv_temp)

            predicted_answers, _ = self.model.generate(
                image,
                texts,
                max_new_tokens=self.config.run.max_new_tokens,
                do_sample=False,
                calc_probs=False
            )

            for p_answer, question_id, ans in zip(predicted_answers, question_ids, answers):
                if not isinstance(p_answer, str):
                    p_answer = str(p_answer)
                clean_answer = p_answer.replace('#', '')
                p_answer = clean_answer.lower().strip()

                if not isinstance(ans, str):
                    ans = str(ans)
                clean_gt = ans.replace('#', '')
                ans = clean_gt

                self.prepare_for_compute_scores(p_answer, question_id, ans)

            self.save_eval_state(step, self._predictions, self._question_ids, self._ground_truths)
            if is_master():
                self.logger.info(f"Eval step ended: {step}")

        overall_acc, acc_yes_no, acc_number, acc_other = self.compute_vqa_accuracy()
        precision, recall, f1 = self.compute_bertscore()
        bleu = self.compute_bleuscore()

        if is_master():
            after_time = time.time()
            elapsed_time = str(timedelta(seconds=(after_time - before_time)))
            self._log.append(
                f"{precision.item():.6f}\t{recall.item():.6f}\t{f1.item():.6f}\t{bleu.item():.6f}\t"
                f"{overall_acc.item():.6f}\t{acc_yes_no.item():.6f}\t{acc_number.item():.6f}\t{acc_other.item():.6f}\t{elapsed_time}"
            )
            file_path = os.path.join(self.config.run.output_dir, "eval_output.txt")
            file_exists = os.path.exists(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a') as f:
                if not file_exists:
                    f.write("precision\trecall\tf1\tbleu\toverall_acc\tacc_yes_no\tacc_number\tacc_other\ttime\n")
                f.write("\n".join(self._log) + "\n")
            self.logger.info("Eval ended")

        if is_dist():
            dist.barrier()

    def prepare_for_compute_scores(self, prediction, question_id, answer):
        if prediction.strip() == "":
            prediction = "[EMPTY]"
            if is_master():
                self.logger.info("empty detected")
        self._predictions.append(prediction)
        self._ground_truths.append(answer)
        self._question_ids.append(question_id)

    def compute_vqa_accuracy(self):
        evaluator = VQAEval(self._predictions, self._question_ids, self._annotations_paths)
        accuracy, acc_yes_no, acc_number, acc_other = evaluator.evaluate()
        # Tensores no device para consistência
        accuracy = torch.tensor(accuracy, device=self.device)
        acc_yes_no = torch.tensor(acc_yes_no, device=self.device)
        acc_number = torch.tensor(acc_number, device=self.device)
        acc_other = torch.tensor(acc_other, device=self.device)
        return accuracy, acc_yes_no, acc_number, acc_other

    def clean_text(self, text):
        return text.replace("#", "").lower().strip()

    def compute_bertscore(self):
        p, r, f1 = score(self._predictions, self._ground_truths, lang="en")
        # mover para device se necessário
        if p.device.type != self.device.type:
            p = p.to(self.device)
            r = r.to(self.device)
            f1 = f1.to(self.device)
        return p.mean(), r.mean(), f1.mean()

    def compute_bleuscore(self):
        tokens_predictions = [word_tokenize(p) for p in self._predictions]
        tokens_ground_truths = [[word_tokenize(g)] for g in self._ground_truths]
        weights_bleu_1 = (1, 0, 0, 0)
        sc = corpus_bleu(tokens_ground_truths, tokens_predictions, weights=weights_bleu_1, smoothing_function=self._smooth_fn)
        sc = torch.tensor(sc, device=self.device)
        return sc

    def finalize(self):
        pass

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    @staticmethod
    def maybe_add_noise(batch_sample, noise_level):
        if noise_level > 0:
            image = batch_sample["image"]
            noise = torch.randn_like(image) * noise_level  # GAUSSIAN
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

                self._questions_paths = split.questions_paths
                self._annotations_paths = split.annotations_paths

                is_train = True if split.split_name in self.config.run.train_splits else False
                collate_fn = getattr(split, "collater", None)

                sampler = None
                if self.config.run.distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1 and is_dist():
                    sampler = DistributedSampler(
                        split,
                        num_replicas=registry.get("world_size", 1),
                        rank=registry.get("rank", 0),
                        shuffle=is_train
                    )

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

    def create_optimizer(self):
        self.logger.info("Creating the optimizer")
        beta1 = self.config.run.beta1
        beta2 = self.config.run.beta2
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.run.init_lr),
            weight_decay=float(self.config.run.weight_decay),
            betas=(beta1, beta2),
        )

    def build_model(self):
        self.logger.info("Start building the model")
        model_type = registry.get_model_class(self.config.arch)
        model = model_type.from_config(self.config.model)
        model.to(self.device)
        # DDP (se for o caso, deixe para agentes de treinamento)
        return model

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        for conv, text in zip(convs, texts):
            conv.append_message(conv.roles, text)
            conv.append_message(conv.roles[1], None)
        texts = [conv.get_prompt() for conv in convs]
        return texts

    def save_eval_state(self, step, predictions, question_ids, ground_truths):
        if is_master():
            self.logger.info("saving state..")
        state = dict(step=step, predictions=predictions, question_ids=question_ids, ground_truths=ground_truths)
        rank = get_rank()
        os.makedirs(self.config.run.output_dir, exist_ok=True)
        file_path = os.path.join(self.config.run.output_dir, f"eval_output_r{rank}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        if is_master():
            self.logger.info("state saved!")
        if is_dist():
            dist.barrier()

    def load_eval_state(self):
        rank = get_rank()
        file_path = os.path.join(self.config.run.output_dir, f"eval_output_r{rank}.pkl")
        if not os.path.exists(file_path):
            if is_master():
                self.logger.info(f'file not found: {file_path}')
            return None
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        if is_master():
            self.logger.info("eval_state_loaded")
        if is_dist():
            dist.barrier()
        return state
