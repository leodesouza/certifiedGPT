# This code was created by Jeremy Cohen, Elan Rosenfeld, and Zico Kolter 
# for the paper Certified Adversarial Robustness via Randomized Smoothing (https://arxiv.org/abs/1902.02918)
# visit the repo at:https://github.com/locuslab/smoothing


import torch
# from scipy.stats import norm, binom_test
from scipy.stats import norm, binomtest

import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import torch_xla.core.xla_model as xm
import torch_xla.amp as xla_amp
from common.registry import registry
from graphs.models.minigpt4.conversation.conversation import CONV_VISION_LLama2
import torch_xla.test.test_utils as test_utils


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_decoder: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_decoder: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_decoder = base_decoder
        self.num_classes = num_classes
        self.sigma = sigma
        self._device = xm.xla_device()
        self.config = registry.get_configuration_class("configuration")

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_decoder.eval()
        xm.master_print("draw samples of f(x+ epsilon)")
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        xm.master_print(f"Printing counts_selection:{counts_selection}")        
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_decoder.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, batch_sample: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param batch_sample: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """

        question = batch_sample["instruction_input"]
        answers = batch_sample["answer"]
        question_id = batch_sample["question_id"]
        image_id = batch_sample["image_id"]

        xm.master_print(f"QuestionId: {question_id}")
        xm.master_print(f"Question: {question}")
        xm.master_print(f"Answer: {answers}")
        xm.master_print(f"ImageId: {image_id}")

        conv_temp = CONV_VISION_LLama2.copy()
        conv_temp.system = ""
                 
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                                
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                image = batch_sample["image"]
                batch_image = image.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch_image, device=self._device) * self.sigma
                
                batch_question = question * this_batch_size
                questions = self.prepare_texts(batch_question, conv_temp)                
                predictions = []                
                max_tokens = self.config.run.max_new_tokens 

                with xla_amp.autocast(enabled=self.config.run.amp, device=self._device):
                    answers, probs = (self.base_decoder.generate(batch_image + noise, questions, max_new_tokens=max_tokens, do_sample=False))
                
                xm.mark_step()                

                for question, answer, prob, img_id in zip(questions, answers, probs, image_id):
                    result = dict()
                    answer = answer.lower().replace('<unk>', '').strip()
                    result['question'] = question
                    result['answer'] = answer
                    result['prob'] = prob
                    result['image_id'] = int(img_id)
                    predictions.append(result)
            
                xm.master_print(f"predictions: {predictions}")
                xm.master_print(f"probs: {probs}")

                raise Exception("terminou!!!")
                                
                # predicted_tokens = torch.argmax(logits, dim=-1)
                # generated_text = self.base_decoder.llama_tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
                # xm.master_print(f"generated text: {generated_text}")
                # probs = torch.softmax(logits, dim=-1)
                # xm.master_print("calc probs and asign to counts")
                # counts += probs.cpu().numpy().sum(axis=0)

            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(
            conv.roles[0], text) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts
