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
        self.answers = {
            ann["question_id"]:ann for ann in self.annotations["annotations"]            
        }        

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
    
    def evaluate(self):        
        acc_per_question = {}
        acc_per_question_yes_no = {}
        acc_per_question_number = {}
        acc_per_question_other = {}
        for idx, (pred, question_id) in enumerate(zip(self.preds, self.question_ids)):                        
            answers = self.answers.get(question_id)
            answer_type = answers["answer_type"]                                     
            answers = answers["answers"]            
            gt_answers = [
                self.normalize_vqa_answer(ann["answer"]) 
                for ann in answers                 
            ]            
                        
            if not gt_answers:
                acc = 0.0
            else:
                norm_pred = self.normalize_vqa_answer(pred)                
                acc = self.compute_accuracy(norm_pred, gt_answers)                                    
            acc_per_question[idx] = acc

            if answer_type ==  "yes/no":
                acc_per_question_yes_no[idx] = self.compute_soft_accuracy(norm_pred, gt_answers)
            
            if answer_type == "number":
                acc_per_question_number[idx] = self.compute_soft_accuracy(norm_pred, gt_answers)

            if answer_type == "other":
                acc_per_question_other[idx] = self.compute_soft_accuracy(norm_pred, gt_answers)
            
            if acc < 1.0:
                print(f"[FAIL] pred: '{norm_pred}' vs. gts: {gt_answers}: question_id: {question_id}")

        if not acc_per_question:
            return {"overall": 0.0}
        print(f"acc_per_question: {len(acc_per_question)}")
        
        overall_acc = 100.0 * sum(acc_per_question.values()) / len(acc_per_question)
        acc_yes_no = ( 
            100.0 * sum(acc_per_question_yes_no.values()) / len(acc_per_question_yes_no)
            if acc_per_question_yes_no else 0.0                      
        )
        acc_number = (
            100.0 * sum(acc_per_question_number.values()) / len(acc_per_question_number)
            if acc_per_question_number else 0.0
        )

        acc_other = (
            100.0 * sum(acc_per_question_other.values()) / len(acc_per_question_other)
            if acc_per_question_other else 0.0
        )        

        return overall_acc, acc_yes_no, acc_number, acc_other
         
    def is_close_match(self, a, b, threshold=0.7):
        return SequenceMatcher(None, a, b).ratio() >= threshold
    
    def compute_soft_accuracy(self, pred, gts):        
        ''''
        Check https://arxiv.org/pdf/2211.09699 paper to get insights on how to implement the soft
        VQA accuracy. Code below is just an inspiration of the metric using soft matching.
        '''
        matchs = sum([1 for gt in gts if self.is_close_match(pred, gt)])        
        return min(1.0, matchs /3)        

    def compute_accuracy(self, pred, gts):                
        matchs = sum([1 for gt in gts if pred == gt])        
        return min(1.0, matchs /3)
    
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText
    
    def normalize_vqa_answer(self, ans):
        if not ans:
            return ""
        ans = ans.replace('\n', ' ').replace('\t', ' ').strip().lower()
        ans = self.processPunctuation(ans)
        ans = self.processDigitArticle(ans)
        ans = ans.strip()

        words = ans.split()
        cleaned = []
        seen = set()
        for w in words:
            if w not in seen:
                cleaned.append(w)
                seen.add(w)
        
        return " ".join(cleaned)
        
        
            
