# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from the "MiniGPT-4" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/Vision-CAIR/MiniGPT-4
#

import transformers

transformers.logging.set_verbosity_error()

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
# Torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Pytorch XLA

from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from common.vqa_tools.vqa import VQA
from common.vqa_tools.vqa_eval import VQAEval

from agents.base import BaseAgent
from common.metrics import TPUMetrics
from common.registry import registry
import torch_xla.test.test_utils as test_utils
from graphs.models.minigpt4.conversation.conversation import CONV_VISION_LLama2

from bert_score import score 
import datetime
from time import time

import nltk
nltk.download('punkt_tab')

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 
from nltk.tokenize import word_tokenize

# rank and world size are inferred from XLA Device
# source: https://github.com/pytorch/xla/
dist.init_process_group(backend='xla', init_method='xla://')

@registry.register_agent("image_text_eval")
class MiniGPT4EvalAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.start_step = 0
        self._device = xm.xla_device()
        self._model = self.build_model()
        self._tpu_metrics = TPUMetrics()
        self._questions_paths = None
        self._annotations_paths = None
        self._log = []
        self._smooth_fn = SmoothingFunction().method1                
        self._prediction_dict  = {}
        self._groud_truth_answer_dict = {}

    def run(self):
        try:

            self.logger.info("Creating the dataloaders")
            self._dataloaders = self.create_dataloaders()

            if xm.is_master_ordinal():
                if self.config.run.noise_level > 0:
                    xm.master_print(f"Noise level: {self.config.run.noise_level} will be applied to the image inputs")
                else:
                    xm.master_print("No noise will be applied to the image inputs")

            self.load_finetuned_model(self._model)
            self.eval(self._dataloaders)            

        except Exception as e:
            xm.master_print(f"Error on agent run: {test_utils.now()}. Details: {e}")

    @torch.no_grad()
    def eval(self, dataloader):        
        val_loader = dataloader["val"]
        val_loader = pl.MpDeviceLoader(val_loader, self.device)

        if len(val_loader) == 0:
            return float("inf")

        conv_temp = CONV_VISION_LLama2.copy()
        conv_temp.system = ""

        xm.master_print(f"Eval started: {(test_utils.now())}")        
        before_time = time()        
        self.model.eval()        
        for step, batch_sample in enumerate(val_loader):

            if step > 0:
                continue

            xm.master_print(f"Eval step: {step} - {(test_utils.now())}")            
            self.maybe_add_noise(batch_sample, self.config.run.noise_level)
            
            image = batch_sample["image"]
            question_ids = batch_sample["question_id"]            
            questions = batch_sample["instruction_input"]            
            ground_truth_answers = batch_sample["answer"]
            answers_type = batch_sample["answer_type"]
            
            xm.master_print("batch read") 
            texts = self.prepare_texts(questions, conv_temp)

            predicted_answers, _ = (self.model.
                       generate(image, texts, max_new_tokens=self.config.run.max_new_tokens, do_sample=False, calc_probs=False))
            xm.mark_step()

            for question_id, p_answer, g_answer, ans_type in zip(question_ids, predicted_answers, ground_truth_answers, answers_type):
                if isinstance(p_answer, str):
                    clean_answer = p_answer.replace('#','')
                    p_answer = clean_answer.lower().replace('<unk>','').strip()
                
                if isinstance(g_answer, str):
                    clean_answer = g_answer.replace('#','')
                    g_answer = clean_answer.lower().replace('<unk>','').strip()
                xm.master_print("prepare_for_compute_scores") 
                self.prepare_for_compute_scores(question_id, p_answer, g_answer)   
                xm.master_print("prepare_for_compute_scores 2") 
            

        xm.master_print("computing vqa accuracy")        
        overall, per_question = self.compute_vqa_accuracy()
        xm.master_print(f"overall: {overall}")
        xm.master_print(f"per_question: {per_question}")

        xm.master_print("computing bert score")        
        precision, recall, f1 = self.compute_bertscore()

        xm.master_print("computing blue score")        
        bleu = self.compute_bleuscore()
        
        xm.master_print("mesh_reduce") 
        global_precision = xm.mesh_reduce("precision", precision.item(), lambda x: sum(x) / len(x)) 
        global_recall = xm.mesh_reduce("recall", recall.item(), lambda x: sum(x) / len(x)) 
        global_f1 = xm.mesh_reduce("f1", f1.item(), lambda x: sum(x) / len(x))
        global_bleu_score = xm.mesh_reduce("blue", bleu.item(), lambda x: sum(x) / len(x))        
                                                
        if xm.is_master_ordinal():
                       
            after_time = time()
            elapsed_time = str(datetime.timedelta(seconds=(after_time - before_time)))
        
            self._log.append(f"{global_precision}\t{global_recall}\t{global_f1}\t{global_bleu_score}\t{elapsed_time}")
            file_path = os.path.join(self.config.run.output_dir,"eval_output.txt")
            file_exists = os.path.exists(file_path)

            with open(file_path, 'a') as f:
                if not file_exists:
                    f.write("precision\trecall\tf1\tbleu\ttime\n")
                f.write("\n".join(self._log) + "\n")

        xm.master_print(f"Eval ended: {(test_utils.now())}")
    
    def prepare_for_compute_scores(self, question_id, prediction, groud_truth_answer):
        
        if prediction.strip() == "":
            xm.master_print("empty prediction detected")
            prediction = "[EMPTY]"

        self._prediction_dict[question_id.item()].append(prediction)                                        
        self._groud_truth_answer_dict[question_id.item()].append(groud_truth_answer)                

    def compute_vqa_accuracy(self):
        xm.master_print(f"self._groud_truth_answer_dict:{self._groud_truth_answer_dict}")
        xm.master_print(f"self._prediction_dict:{self._prediction_dict}")

        evaluator = VQAEval(self._groud_truth_answer_dict, self._prediction_dict)
        evaluator.evaluate()
        overall_acuracy = evaluator.get_accuracy()
        overall_acuracy = torch.tensor(overall_acuracy, device=self.device)
        per_question_accuracy = evaluator.get_per_question_accuracy()
        per_question_accuracy = torch.tensor(per_question_accuracy, device=self.device)

        return overall_acuracy, per_question_accuracy 
        
    def clean_text(self, text):
        return text.replace("#", "").lower().replace("<unk>","").strip()
    
    def compute_bertscore(self):                
        predictions = [p for p in self._prediction_dict.values()]
        ground_truths = [g for g in self._groud_truth_answer_dict.values()]

        p, r, f1 = score(predictions, ground_truths, lang="en")                
        xm.mark_step()
        
        p = p.to(self.device)
        r = r.to(self.device)
        f1 = f1.to(self.device)
                
        return p.mean(), r.mean(), f1.mean()
    
    def compute_bleuscore(self):               

        predictions = [p for p in self._prediction_dict.values()]
        ground_truths = [g for g in self._groud_truth_answer_dict.values()]

        tokens_predictions = [word_tokenize(p) for p in predictions]
        tokens_ground_truths = [[word_tokenize(g)] for g in ground_truths]
        weights_bleu_1 = (1,0,0,0)        
        score = corpus_bleu(tokens_ground_truths, tokens_predictions, weights=weights_bleu_1, smoothing_function=self._smooth_fn)
        score = torch.tensor(score, device=self.device)        
        return score


    def finalize(self):
        pass

    @classmethod
    def setup_agent(cls, **kwargs):
        return cls()

    @staticmethod
    def maybe_add_noise(batch_sample, noise_level):

        if noise_level > 0:
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
                xm.master_print(f"creating split: {split.split_name}")        
                num_records = len(split)
                if num_records >= 0:
                    self.logger.info(
                        "Loaded {} records for split {}".format(num_records, split.split_name)
                    )

                self._questions_paths = split.questions_paths
                self._annotations_paths = split.annotations_paths

                is_train = (
                    True if split.split_name in self.config.run.train_splits else False
                )

                xm.master_print("getattr collater")        
                collate_fn = getattr(split, "collater", None)

                xm.master_print("sampler")        
                sampler = DistributedSampler(
                    split,
                    num_replicas=xr.world_size(),
                    rank=xm.runtime.global_ordinal(),
                    shuffle=True if is_train else False
                ) if self.config.run.distributed and xr.world_size() > 1 else None

                xm.master_print("creating loader")        
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
        return model

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(
            conv.roles[0], text) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
