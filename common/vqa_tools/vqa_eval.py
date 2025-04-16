import re
import json
from difflib import SequenceMatcher

class VQAEval:
    def __init__(self, preds=None, question_ids=None, annotation_path=None):
                
        if preds is None or question_ids is None:
            raise ValueError("Both preds, question_ids must be provided")
        
        if annotation_path is None:
            raise ValueError("annotation must be provided ")
                
        self.preds = preds        
        self.question_ids = question_ids                
        self.annotations = json.load(open(annotation_path[0], 'r'))        
        self.answers = {ann["question_id"]:ann for ann in self.annotations["annotations"]}        
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
    
    def is_close_match(a, b, threshold=0.9):
        return SequenceMatcher(None, a, b).ratio >= threshold
    
    def compute_accuracy(self, pred, gts):
        matchs = sum([1 for gt in gts if self.is_close_match(pred, gt)])
        return min(1.0, matchs /3)
    

    def evaluate(self):        
        acc_per_question = {}
        for idx, (pred, question_id) in enumerate(zip(self.preds, self.question_ids)):                        
            answers = self.answers.get(question_id)            
            answers = answers["answers"]
            print(f"answers: {answers}")
            gt_answers = [
                self.normalize_answer(ann["answer"]) 
                for ann in answers 
                if ann["answer_confidence"] in ["yes","maybe"]
            ]            

            print(f"gt_answers: {gt_answers}")
            if not gt_answers:
                acc = 0.0
            else:
                norm_pred = self.normalize_answer(pred)
                acc = self.compute_accuracy(norm_pred, gt_answers)                                    
            acc_per_question[idx] = acc

        if not acc_per_question:
            return {"overall": 0.0}

        overall_acc = 100.0 * sum(acc_per_question.values()) / len(acc_per_question)
        print(f"overall_acc: {overall_acc}")

        return {
            "overall": overall_acc            
        }
        
            
