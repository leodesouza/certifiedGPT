import re

class VQAEval:
    def __init__(self, gts=None, preds=None):
        """
        gts: dict mapping question_id to a list of ground truth answers
        preds: dict mapping question_id to predicted answer
        """        
        self.gts = gts or {}
        self.preds = preds or {}
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
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = r";/[]\"{}()=+\\_-<>@`?,!"

    def normalize_answer(self, ans):
        """Apply string normalization: lowercase, punctuation, articles, etc."""
        ans = ans.replace('\n', ' ').replace('\t', ' ').strip().lower()

        # Remove punctuations
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

    def evaluate(self):
        acc_per_question = {}

        for qid, gt_answers in self.gts.items():
            pred_ans = self.preds.get(qid, "")
            norm_pred = self.normalize_answer(pred_ans)

            norm_gt = [self.normalize_answer(gt) for gt in gt_answers]
            matching = sum([norm_pred == gt for gt in norm_gt])
            acc = min(1.0, matching / 3.0)
            acc_per_question[qid] = acc

        self.evalQA = acc_per_question
        self.accuracy = {"overall": 100.0 * sum(acc_per_question.values()) / len(acc_per_question)}

    def get_accuracy(self):
        return self.accuracy

    def get_per_question_accuracy(self):
        return self.evalQA
