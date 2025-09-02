# build_vqa_vocab_official.py
import os, json
from collections import Counter
from typing import List, Dict
from vqa_eval import VQAEval  # your current file

def build_vocab_with_official_normalization(qa_pairs: List[Dict], top_k: int = 5000, min_freq: int = 2):
    """
    Build vocabulary using the exact same normalization as your VQA evaluation.
    Uses normalize_vqa_answer() method for 100% consistency.
    
    Args:
        qa_pairs: List of dicts with keys 'question' and 'answers' (list of strings)
        top_k: Maximum number of answers to keep in vocabulary
        min_freq: Minimum frequency threshold for including answers
        
    Returns:
        tuple: (vocab_list, counter) where vocab_list is sorted by descending frequency
    """
    # Create instance only to access normalization methods
    evaluator = VQAEval(preds=["dummy"], question_ids=[0], annotation_path=["dummy_path"])
    
    counter = Counter()
    
    for ex in qa_pairs:
        for answer in ex["answers"]:
            # Use the official method from your evaluator
            normalized = evaluator.normalize_vqa_answer(answer)
            if normalized:  # not empty after normalization
                counter[normalized] += 1
    
    # Apply min_freq and sort
    items = [(ans, cnt) for ans, cnt in counter.items() if cnt >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))  # descending frequency, lexicographic
    
    # Cut by top_k
    vocab = [ans for ans, _ in items[:top_k]]
    
    return vocab, counter

def save_vocab_json(vocab: List[str], out_path: str):
    """Save vocabulary to JSON format compatible with SmoothV2._load_vocab()"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"answers": vocab}, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab with {len(vocab)} entries to {out_path}")

def load_vqav2_pairs(root: str, splits=("train","val")) -> List[Dict]:
    """
    Load VQAv2 questions+annotations and build qa_pairs.
    
    Expected structure:
      root/questions/v2_OpenEnded_mscoco_{split}2014_questions.json
      root/annotations/v2_mscoco_{split}2014_annotations.json
      
    Args:
        root: Path to VQAv2 dataset root directory
        splits: Tuple of splits to load ('train', 'val')
        
    Returns:
        List of dicts with keys: {question: str, answers: List[str]}
    """
    all_pairs = []
    for split in splits:
        qfile = f"v2_OpenEnded_mscoco_{split}2014_questions.json"
        afile = f"v2_mscoco_{split}2014_annotations.json"
        
        with open(os.path.join(root, "questions", qfile), "r", encoding="utf-8") as f:
            qjson = json.load(f)
        with open(os.path.join(root, "annotations", afile), "r", encoding="utf-8") as f:
            ajson = json.load(f)

        qdict = {q["question_id"]: q["question"] for q in qjson["questions"]}
        for ann in ajson["annotations"]:
            qid = ann["question_id"]
            if qid in qdict:
                question = qdict[qid]
                answers = [a["answer"] for a in ann.get("answers", []) if a.get("answer")]
                if answers:
                    all_pairs.append({"question": question, "answers": answers})
    return all_pairs

if __name__ == "__main__":
    # Configuration paths and hyperparameters
    VQAv2_ROOT = "/path/to/VQAv2"
    OUT_JSON = "/path/to/vocab/vqa_vocab_official_top5k.json"
    TOP_K = 5000
    MIN_FREQ = 2

    # Generate vocabulary using official normalization
    qa_pairs = load_vqav2_pairs(VQAv2_ROOT, splits=("train", "val"))
    vocab, freqs = build_vocab_with_official_normalization(qa_pairs, top_k=TOP_K, min_freq=MIN_FREQ)
    save_vocab_json(vocab, OUT_JSON)

    # Basic logging
    print("Top 20 answers:", vocab[:20])
    print(f"Vocabulary covers {len(vocab)} unique normalized answers")
    print(f"Total answer occurrences: {sum(freqs.values())}")
