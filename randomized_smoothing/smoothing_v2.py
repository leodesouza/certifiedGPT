# Based on: Cohen, Rosenfeld, Kolter — "Certified Adversarial Robustness via Randomized Smoothing"
# Original repo: https://github.com/locuslab/smoothing
# Note: This file contains adaptations for vision–language decoders and GPU inference.
# Changes include: 
# (i) mapping free-text generations to a finite label space Y∪{UNK};
# (ii) selection via majority over n0 and estimation via Clopper–Pearson LCB (p_A) and maximum UCB across competitors (p_B); (iii) certified radius R = (σ/2)(Φ^{-1}(p_A)−Φ^{-1}(p_B));
# (iv) integration with MiniGPT-4 decoding; (v) CUDA/AMP support and removal of XLA-specific code.
import os, json, re
from collections import Counter
from math import ceil
from typing import Union

import numpy as np
import torch
from scipy.stats import norm, binomtest
from statsmodels.stats.proportion import proportion_confint

from common.registry import registry
from graphs.models.minigpt4.conversation.conversation import CONV_VISION_LLama2
from common.vqa_tools.vqa_eval import VQAEval

# GPU autocast
from torch.cuda.amp import autocast


class SmoothV2(object):
    """A smoothed classifier g"""

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_decoder: torch.nn.Module, sigma: float):
        """
        :param base_decoder: a model instance
        :param sigma: the noise level hyperparameter
        """
        self.base_decoder = base_decoder
        self.sigma = sigma

        # device: prefer registry's device if available
        self._device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = registry.get_configuration_class("configuration")
        self.UNK = "UNK"
        self.vocab_set, self.vocab_list = self._load_vocab(self.config.run.vocab_file_path)
        self._vqa_normalizer = VQAEval(preds=["dummy"], question_ids=[0], annotation_path=["dummy_path"])
        

    def _load_vocab(self, vocab_path, add_unk=True):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        ext = os.path.splitext(vocab_path)[1].lower()
        if ext == ".json":
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = (data["answers"] if isinstance(data, dict) and "answers" in data else list(data))
        else:
            with open(vocab_path, "r", encoding="utf-8") as f:
                entries = [ln.strip() for ln in f if ln.strip()]
        entries = [e.strip().lower() for e in entries if isinstance(e, str)]
        dedup, seen = [], set()
        for e in entries:
            if e not in seen:
                seen.add(e)
                dedup.append(e)
        if add_unk and self.UNK not in seen:
            dedup.append(self.UNK)
        return set(dedup), dedup

    def _normalize_vqa(self, s: str) -> str:
        return self._vqa_normalizer.normalize_vqa_answer(s)         

    def _map_to_label(self, s: str) -> str:
        s_norm = self._normalize_vqa(s)
        return s_norm if s_norm in self.vocab_set else self.UNK

    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float, batch_size: int):
        """
        Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the answer returned by this method will equal g(x), and g's prediction will
        be robust within an L2 ball of radius R around x.

        :param x: the input dict with keys ["image","instruction_input"]
        :param n0: selection sample size
        :param n: estimation sample size
        :param alpha: failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted answer, certified radius, was_unk_top1). If abstaining, returns (ABSTAIN, 0.0, flag).
        """
        self.base_decoder.eval()

        # Selection by majority label
        sample_for_selection = self._sample_noise(x, n0, batch_size, "selection")
        print(f"sample_for_selection: {sample_for_selection}")
        labels_sel = [lab for lab, _ in sample_for_selection]
        if len(labels_sel) == 0:
            return SmoothV2.ABSTAIN, 0.0, False

        tA = Counter(labels_sel).most_common(1)[0][0]
        top1_is_unk = (tA == self.UNK)

        # Estimation counts
        sample_for_estimation = self._sample_noise(x, n, batch_size, "estimation")
        labels_est = [lab for lab, _ in sample_for_estimation]
        counts = Counter(labels_est)
        nA = counts.get(tA, 0)

        # Clopper-Pearson bounds (LCB/UCB)
        def LCB(k, N, a):
            return proportion_confint(k, N, alpha=2 * a, method="beta")[0] if N > 0 else 0.0

        def UCB(k, N, a):
            return proportion_confint(k, N, alpha=2 * a, method="beta")[1] if N > 0 else 1.0

        pA = LCB(nA, n, alpha)
        competitors = [c for c in counts.keys() if c != tA]
        pB = max([UCB(counts.get(c, 0), n, alpha) for c in competitors], default=0.0)

        if top1_is_unk or pA <= 0.5:
            return SmoothV2.ABSTAIN, 0.0, top1_is_unk

        radius = 0.5 * self.sigma * (norm.ppf(pA) - norm.ppf(pB))
        if radius <= 0:
            return SmoothV2.ABSTAIN, 0.0, top1_is_unk

        return tA, float(radius), top1_is_unk

    def predict(self, x: torch.Tensor, n: int, alpha: float, batch_size: int) -> Union[str, int]:
        """
        Monte Carlo algorithm for evaluating the prediction of g at x. With probability at least 1 - alpha,
        the answer returned by this method will equal g(x).
        Returns the predicted answer, or ABSTAIN.
        """
        self.base_decoder.eval()

        sample_for_estimation = self._sample_noise(x, n, batch_size, "estimation")
        labels = [lab for lab, _ in sample_for_estimation]
        if len(labels) == 0:
            return SmoothV2.ABSTAIN

        counts = Counter(labels).most_common(2)
        if len(counts) == 1:
            return counts

        (lab1, count1), (lab2, count2) = counts, counts[1]
        if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
            return SmoothV2.ABSTAIN
        return lab1

    def _sample_noise(self, batch_sample: dict, num: int, batch_size: int, sample_type: str = "estimation"):
        """
        Sample the base decoder's prediction under noisy corruptions of the input x.
        :param batch_sample: dict with keys ["image","instruction_input"]
        :param num: number of samples to collect
        :param batch_size: per-iteration batch
        :return: np.array of (label, prob) pairs
        """
        question = batch_sample["instruction_input"]
        conv_temp = CONV_VISION_LLama2.copy()
        conv_temp.system = ""        
        predictions = []

        with torch.no_grad():
            self.logger.info(f"Generating sample for {sample_type}")
            step = 1
            for _ in range(ceil(num / batch_size)):
                print(f"remaining {num}")
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                print("batch_sample[image]")
                image = batch_sample["image"].to(self._device)
                batch_image = image.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch_image, device=self._device) * self.sigma
                noisy_image_batch = batch_image + noise

                print("question * this_batch_size")
                batch_question = question * this_batch_size
                print("prepare_texts")
                questions = self.prepare_texts(batch_question, conv_temp)
                print(f"max_new_tokens: {self.config.run.max_new_tokens}")
                max_tokens = self.config.run.max_new_tokens
                
                print("autocast")
                with autocast(enabled=bool(getattr(self.config.run, "amp", False))):
                    answers, probs = self.base_decoder.generate(
                        noisy_image_batch,
                        questions,
                        max_new_tokens=max_tokens,
                        do_sample=False
                    )
                print(f"_map_to_label {answer}")
                for answer, prob in zip(answers, probs):
                    label = self._map_to_label(answer)
                    predictions.append((label, float(prob)))

                self.logger.info(f"Sample: {step} of size: {this_batch_size}")
                step += 1

        predictions = np.array(predictions, dtype=object)
        return predictions

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a Bernoulli proportion (Clopper-Pearson)."""
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        for conv, text in zip(convs, texts):
            conv.append_message(conv.roles, text)
            conv.append_message(conv.roles[1], None)
        texts = [conv.get_prompt() for conv in convs]
        return texts

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger
