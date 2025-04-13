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

import pickle

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
        self._predictions = []
        self._ground_truths = []
        self._anwers_type = []

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
            self.logger.info(f"Error on agent run: {test_utils.now()}. Details: {e}")

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
        # state = self.load_eval_state()       
                
        for step, batch_sample in enumerate(val_loader):
            
            xm.master_print(f"Eval step: {step} - {(test_utils.now())}")            
            self.maybe_add_noise(batch_sample, self.config.run.noise_level)
            
            image = batch_sample["image"]            
            questions = batch_sample["instruction_input"]            
            ground_truth_answers = batch_sample["answer"]
            answers_type = batch_sample["answer_type"]
                        
            texts = self.prepare_texts(questions, conv_temp)

            predicted_answers, _ = (self.model.
                       generate(image, texts, max_new_tokens=self.config.run.max_new_tokens, do_sample=False, calc_probs=False))
            xm.mark_step()

            for p_answer, g_answer, answer_type  in zip(predicted_answers, ground_truth_answers, answers_type):
                if not isinstance(p_answer, str):
                    p_answer = str(p_answer)                
                clean_answer = p_answer.replace('#','')
                p_answer = clean_answer.lower().replace('<unk>','').strip()
                                
                if not isinstance(g_answer, str):
                    g_answer = str(g_answer)
                clean_answer = g_answer.replace('#','')
                g_answer = clean_answer.lower().replace('<unk>','').strip()                
                self.prepare_for_compute_scores(p_answer, g_answer, answer_type)   

            self.save_eval_state(step, self._predictions, self._ground_truths, self._anwers_type)            
                                    
        accuracy = self.compute_vqa_accuracy()          
        precision, recall, f1 = self.compute_bertscore()
        bleu = self.compute_bleuscore()

        xm.master_print(f"accuracy: {accuracy}. precision: {precision}. recall: {recall}. f1: {f1}, bleu: {bleu}")        
                                
        global_precision = xm.mesh_reduce("precision", precision.item(), lambda x: sum(x) / len(x)) 
        global_recall = xm.mesh_reduce("recall", recall.item(), lambda x: sum(x) / len(x)) 
        global_f1 = xm.mesh_reduce("f1", f1.item(), lambda x: sum(x) / len(x))
        global_bleu_score = xm.mesh_reduce("blue", bleu.item(), lambda x: sum(x) / len(x))        
        global_accuracy = xm.mesh_reduce("accuracy", accuracy.item(), lambda x: sum(x) / len(x))        
                                                
        if xm.is_master_ordinal():
                       
            after_time = time()
            elapsed_time = str(datetime.timedelta(seconds=(after_time - before_time)))
        
            self._log.append(f"{global_precision}\t{global_recall}\t{global_f1}\t{global_bleu_score}\t{global_accuracy}\t{elapsed_time}")
            file_path = os.path.join(self.config.run.output_dir,"eval_output.txt")
            file_exists = os.path.exists(file_path)

            with open(file_path, 'a') as f:
                if not file_exists:
                    f.write("precision\trecall\tf1\tbleu\taccuracy\ttime\n")
                f.write("\n".join(self._log) + "\n")

        xm.master_print(f"Eval ended: {(test_utils.now())}")
    
    def prepare_for_compute_scores(self, prediction, groud_truth_answer, answer_type):
        
        if prediction.strip() == "":            
            prediction = "[EMPTY]"
        
        self._predictions.append(prediction)
        self._ground_truths.append(groud_truth_answer)
        self._anwers_type.append(answer_type)        
        
    def compute_vqa_accuracy(self):        
        evaluator = VQAEval(self._ground_truths, self._predictions, self._anwers_type)
        evaluator.evaluate()
        overall_acuracy = evaluator.get_accuracy()        
        overall_acuracy = torch.tensor(overall_acuracy, device=self.device)        

        return overall_acuracy
        
    def clean_text(self, text):
        return text.replace("#", "").lower().replace("<unk>","").strip()
    
    def compute_bertscore(self):                
        
        p, r, f1 = score(self._predictions, self._ground_truths, lang="en")                
        xm.mark_step()
        
        p = p.to(self.device)
        r = r.to(self.device)
        f1 = f1.to(self.device)
                
        return p.mean(), r.mean(), f1.mean()
    
    def compute_bleuscore(self):                       
        tokens_predictions = [word_tokenize(p) for p in self._predictions]
        tokens_ground_truths = [[word_tokenize(g)] for g in self._ground_truths]
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
                
                collate_fn = getattr(split, "collater", None)                
                sampler = DistributedSampler(
                    split,
                    num_replicas=xr.world_size(),
                    rank=xm.runtime.global_ordinal(),
                    shuffle=True if is_train else False
                ) if self.config.run.distributed and xr.world_size() > 1 else None
                
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
    
    def save_eval_state(self, step, predictions, ground_truths, answers_type):        
        xm.master_print("saving state..")   
        state = dict()
        state["step"] = step
        state["predictions"] = predictions
        state["ground_truths"] = ground_truths
        state["answer_type"] = answers_type
        rank = xm.runtime.global_ordinal()
        file_path = os.path.join(self.config.run.output_dir,f"eval_output_{rank}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        xm.master_print("state saved!")   
        xm.rendezvous("eval_state_saved")   
        
        # xm.master_print("Syncronizing in all tpus...")   
        # xm.rendezvous("eval_state_saved")   
        # xm.master_print("Syncronizing completed")   

    def load_eval_state(self):
        rank = xm.runtime.global_ordinal()
        file_path = os.path.join(self.config.run.output_dir, f"eval_output_{rank}.pkl")
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        xm.master_print("eval_state_loaded")   
        xm.rendezvous("eval_state_loaded")
        return state




