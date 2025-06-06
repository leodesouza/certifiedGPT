from common.visualize import save_image
import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from common.registry import registry
from graphs.models.minigpt4.conversation.conversation import CONV_VISION_Vicuna0

from sentence_transformers import SentenceTransformer, util


class Smooth(object):
    """A smoothed classifier g """

    ABSTAIN = -1

    def __init__(self, base_decoder: torch.nn.Module, sigma: float):
        self.base_decoder = base_decoder
        self.sigma = sigma
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = registry.get_configuration_class("configuration")
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        self.base_decoder.eval()

        sample_for_selection = self._sample_noise(x, n0, batch_size, "selection")
        probs_selection = np.array(sample_for_selection[:, 1], dtype=float)
        pAHat = probs_selection.argmax().item()
        text = sample_for_selection[pAHat][0]

        sample_for_estimation = self._sample_noise(x, n, batch_size, "estimation")
        nA = sum(1 for row in sample_for_estimation if row[0] == text)

        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return text, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int):
        self.base_decoder.eval()
        sample_for_estimation = self._sample_noise(x, n, batch_size)
        
        probs_selection = np.array(sample_for_estimation[:, 1], dtype=float)
        top2 = probs_selection.argsort()[::-1][:2]
        
        text1 = sample_for_estimation[top2[0]][0]
        text2 = sample_for_estimation[top2[1]][0]
                                        
        count1 = sum(1 for row in sample_for_estimation if row[0] == text1)
        count2 = sum(1 for row in sample_for_estimation if row[0] == text2)
        
        # binom_test > alpha (non-significant): the difference in occurrences of text1 and text2 is not statistically significant         
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            #reject the null hypothesis and conclude that
            top = top2[0]
            text = sample_for_estimation[top][0]
            return text
    
    def is_similiar(self, text1, text2):
        similarity_threshold = 0.9
        embp = self.sentence_transformer.encode(text1, convert_to_tensor=True)
        embt = self.sentence_transformer.encode(text2, convert_to_tensor=True)                                                            
        similarity = util.cos_sim(embp, embt)
        similarity_score = similarity[0][0].item()                    
        return similarity_score >= similarity_threshold
                

    def _sample_noise(self, batch_sample: torch.tensor, num: int, batch_size, sample_type="estimation"):
        
        print('reading batch_sample  instruction_input')
        question = batch_sample["instruction_input"]
        print(question)
        raise ValueError("foda se")
        
        print(f'_sample_noise ============ instruction_input: {question}')
        
        conv_temp = CONV_VISION_Vicuna0.copy()
        conv_temp.system = ""

        with torch.no_grad():
            predictions = []
            # self.logger.info(f"Generating sample for {sample_type}")
            step = 1
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                self.logger.info(f"Sample: {step} of size: {this_batch_size}")
                
                image = batch_sample["image"].to(self._device)
                batch_image = image.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch_image, device=self._device) * self.sigma
                noisy_image_batch = batch_image + noise

                batch_question = question * this_batch_size
                questions = self.prepare_texts(batch_question, conv_temp)
                max_tokens = self.config.run.max_new_tokens

                # Removed `xla_amp.autocast` and used PyTorch's native autocast
                with torch.cuda.amp.autocast(enabled=self.config.run.amp):
                    answers, probs = self.base_decoder.generate(
                        noisy_image_batch, questions, max_new_tokens=max_tokens, do_sample=False
                    )

                for answer, prob in zip(answers, probs):                    
                    answer = answer.lower().replace('<unk>', '').strip()
                    clean_answer = answer.replace('#', '')
                    predictions.append((clean_answer, prob))
                step += 1
            
            predictions = np.array(predictions, dtype=object)                
            return predictions

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def prepare_texts(self, texts, conv_temp):
        convs = [conv_temp.copy() for _ in range(len(texts))]
        [conv.append_message(conv.roles[0], text) for conv, text in zip(convs, texts)]
        [conv.append_message(conv.roles[1], None) for conv in convs]
        texts = [conv.get_prompt() for conv in convs]
        return texts

    @property
    def logger(self):
        logger = registry.get_configuration_class("logger")
        return logger        