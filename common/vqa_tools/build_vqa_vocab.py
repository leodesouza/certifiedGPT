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

def load_vqav2_pairs(root: str = None, 
                     splits=("train", "val"), 
                     question_files: Dict[str, str] = None,
                     annotation_files: Dict[str, str] = None,
                     custom_paths: Dict[str, Dict[str, str]] = None) -> List[Dict]:
    """
    Load VQAv2 questions+annotations and build qa_pairs with flexible file specification.
    
    Args:
        root: Path to VQAv2 dataset root directory (used with default structure)
        splits: Tuple of splits to load ('train', 'val', 'test', etc.)
        question_files: Dict mapping split -> question filename (overrides default naming)
        annotation_files: Dict mapping split -> annotation filename (overrides default naming)  
        custom_paths: Dict mapping split -> {'questions': full_path, 'annotations': full_path}
                     (overrides root-based paths entirely)
        
    Expected default structure:
      root/questions/v2_OpenEnded_mscoco_{split}2014_questions.json
      root/annotations/v2_mscoco_{split}2014_annotations.json
      
    Example usage:
        # Default usage
        pairs = load_vqav2_pairs("/path/to/VQAv2")
        
        # Custom files within standard structure
        pairs = load_vqav2_pairs("/path/to/VQAv2", 
                                question_files={"train": "custom_train_questions.json"})
        
        # Completely custom paths
        pairs = load_vqav2_pairs(custom_paths={
            "train": {
                "questions": "/custom/path/train_q.json",
                "annotations": "/custom/path/train_a.json"
            }
        })
        
    Returns:
        List of dicts with keys: {question: str, answers: List[str]}
    """
    all_pairs = []
    
    # Default file naming convention
    default_question_files = {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json"
    }
    
    default_annotation_files = {
        "train": "v2_mscoco_train2014_annotations.json", 
        "val": "v2_mscoco_val2014_annotations.json",
        "test": "v2_mscoco_test2015_annotations.json",
        "test-dev": "v2_mscoco_test-dev2015_annotations.json"
    }
    
    for split in splits:
        print(f"Loading {split} split...")
        
        # Determine file paths
        print(f"Determine file paths")
        if custom_paths and split in custom_paths:
            # Use completely custom paths
            qfile_path = custom_paths[split]["questions"]
            print(f"questin path: {qfile_path}")

            afile_path = custom_paths[split]["annotations"]
            print(f"annotation path: {afile_path}")
        else:
            # Use root + custom or default filenames
            if root is None:
                raise ValueError("Either 'root' or 'custom_paths' must be provided")
                
            qfile = (question_files or {}).get(split, default_question_files.get(split))
            afile = (annotation_files or {}).get(split, default_annotation_files.get(split))
            
            if qfile is None or afile is None:
                raise ValueError(f"No default file mapping for split '{split}'. "
                               f"Please provide custom file names.")
            
            qfile_path = os.path.join(root, "questions", qfile)
            afile_path = os.path.join(root, "annotations", afile)
        
        # Check file existence
        if not os.path.exists(qfile_path):
            print(f"Warning: Question file not found: {qfile_path}")
            continue
        if not os.path.exists(afile_path):
            print(f"Warning: Annotation file not found: {afile_path}")
            continue
            
        # Load files
        try:
            with open(qfile_path, "r", encoding="utf-8") as f:
                qjson = json.load(f)
            with open(afile_path, "r", encoding="utf-8") as f:
                ajson = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error loading files for {split}: {e}")
            continue
            
        # Build question dictionary
        qdict = {q["question_id"]: q["question"] for q in qjson.get("questions", [])}
        print(f"Loaded {len(qdict)} questions for {split}")
        
        # Process annotations
        annotations = ajson.get("annotations", [])
        pairs_count = 0
        
        for ann in annotations:
            qid = ann.get("question_id")
            if qid and qid in qdict:
                question = qdict[qid]
                answers = [a["answer"] for a in ann.get("answers", []) if a.get("answer")]
                if answers:
                    all_pairs.append({
                        "question": question, 
                        "answers": answers,
                        "split": split,  # Add split info for debugging
                        "question_id": qid  # Add question_id for traceability
                    })
                    pairs_count += 1
        
        print(f"Created {pairs_count} question-answer pairs for {split}")
    
    print(f"Total pairs loaded: {len(all_pairs)}")
    return all_pairs

# Convenience function for common custom usage
def load_custom_vqa_files(question_path: str, annotation_path: str) -> List[Dict]:
    """
    Load VQA pairs from specific question and annotation files.
    
    Args:
        question_path: Full path to questions JSON file
        annotation_path: Full path to annotations JSON file
        
    Returns:
        List of question-answer pairs
    """
    return load_vqav2_pairs(
        custom_paths={
            "custom": {
                "questions": question_path,
                "annotations": annotation_path
            }
        },
        splits=("custom",)
    )


if __name__ == "__main__":
    # Configuration paths and hyperparameters
    # VQAv2_ROOT = "/path/to/VQAv2"
    # VQAv2_ROOT = "/home/swf_developer/storage/datasets/vqav2"
    OUT_JSON = "/home/swf_developer/storage/datasets/vqav2/vocab/vqa_vocab_official_top5k.json"
    TOP_K = 5000
    MIN_FREQ = 2

    question_path = "/home/swf_developer/storage/datasets/vqav2/questions/val/v2_OpenEnded_mscoco_val2014_questions.json"
    annotation_path = "/home/swf_developer/storage/datasets/vqav2/annotations/val/sample_v2_mscoco_val2014_annotations__2.json"
    # Generate vocabulary using official normalization
    qa_pairs = load_custom_vqa_files(question_path, annotation_path)
    vocab, freqs = build_vocab_with_official_normalization(qa_pairs, top_k=TOP_K, min_freq=MIN_FREQ)
    save_vocab_json(vocab, OUT_JSON)

    # Basic logging
    print("Top 20 answers:", vocab[:20])
    print(f"Vocabulary covers {len(vocab)} unique normalized answers")
    print(f"Total answer occurrences: {sum(freqs.values())}")
