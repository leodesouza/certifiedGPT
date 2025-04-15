import re

class VQAEval:
    def __init__(self, gts=None, preds=None, answers_type=None):
        """
        gts: list of ground truth answers (strings)
        preds: list of predicted answers (strings)
        """
        if gts is None or preds is None or answers_type is None:
            raise ValueError("Both gts, preds and answers_type must be provided")
        if len(gts) != len(preds):
            raise ValueError("Length of ground truth list and prediction list must match")

        self.gts = gts
        self.preds = preds
        self.answers_type = answers_type
        self.accuracy = {}
        self.evalQA = {}

        self.manualMap = {
            "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        self.articles = ["a", "an", "the"]
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
            "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
            "im": "i'm", "ive": "i've", "isnt": "isn't", "its": "it's", "lets": "let's",
            "mightve": "might've", "mustve": "must've", "shes": "she's", "shouldve": "should've",
            "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyd": "they'd",
            "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "werent": "weren't",
            "whatre": "what're", "whats": "what's", "whos": "who's", "wont": "won't",
            "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd", "youre": "you're",
            "youve": "you've"
        }
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(,)(\d)")
        self.punct = r";/[]\"{}()=+\\_-<>@`?,!"
    
    def normalize_answer(self, ans):
        ans = ''.join(ans)
        ans = ans.replace('\n', ' ').replace('\t', ' ').strip().lower()

        for p in self.punct:
            ans = ans.replace(p, '' if p != '.' else ' ')
        ans = self.periodStrip.sub("", ans)
        ans = self.commaStrip.sub(r"\1\3", ans)

        words = ans.split()
        cleaned = []
        for word in words:
            word = self.manualMap.get(word, word)
            if word not in self.articles:
                cleaned.append(self.contractions.get(word, word))
        return " ".join(cleaned)
    
    # def _reduce_repeats(self, ans):
    #     tokens = ans.strip().split()
    #     if all(token == tokens[0] for token in tokens):
    #         return tokens[0]
    #     return ans

    def evaluate(self):        
        acc_per_question = {}
        for idx, (gt, pred, ans_type) in enumerate(zip(self.gts, self.preds, self.answers_type)):                        
            norm_gt = self.normalize_answer(gt)
            norm_pred = self.normalize_answer(pred)
            acc = 1.0 if norm_gt == norm_pred else 0.0
            acc_per_question[idx] = acc
                        
        return {
            "overall": 100.0 * sum(acc_per_question.values()) / len(acc_per_question),
            "yes/no": 100.0 * sum(acc_per_question[idx] for idx in range(len(self.answers_type)) if self.answers_type[idx] == "yes/no") / self.answers_type.count("yes/no"),
            "number": 100.0 * sum(acc_per_question[idx] for idx in range(len(self.answers_type)) if self.answers_type[idx] == "number") / self.answers_type.count("number"),
            "other": 100.0 * sum(acc_per_question[idx] for idx in range(len(self.answers_type)) if self.answers_type[idx] == "other") / self.answers_type.count("other"),
        }
        
            
