import os, json, re
from collections import Counter
from typing import List, Dict


def vqa_normalize(s: str) -> str:
    """
    Basic VQAv2-style normalization:
    - lowercase
    - strip punctuation
    - collapse multiple spaces
    - remove English articles (a, an, the)
    - map number words to digits (0..20)
    """
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)          # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()      # collapse spaces
    s = re.sub(r"\b(a|an|the)\b", " ", s)   # remove articles
    num_map = {
        "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
        "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
        "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15",
        "sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19","twenty":"20"
    }
    tokens = [num_map.get(tok, tok) for tok in s.split()]
    return " ".join(tokens)


def canonicalize(ans_norm: str, q_norm: str) -> str:
    """
    Optional question-aware canonicalization:
    - collapse yes/no
    - extract first integer if present
    - prefer colors when question starts with 'what color/colour'
    - truncate by common prepositions to keep a head noun phrase
    """
    # yes/no
    if ans_norm.startswith("yes"): return "yes"
    if ans_norm.startswith("no"):  return "no"
    # number
    m = re.search(r"\b\d+\b", ans_norm)
    if m: return m.group(0)
    # color
    if q_norm.startswith(("what color", "what colour")):
        colors = {"red","blue","green","black","white","gray","grey","brown","yellow","orange","pink","purple"}
        for tok in ans_norm.split():
            if tok in colors:
                return tok
    # truncate at prepositions to get a head noun phrase
    for prep in (" on ", " in ", " at ", " with ", " by ", " of "):
        idx = ans_norm.find(prep)
        if idx > 0:
            head = ans_norm[:idx].strip()
            if head:
                return head
    return ans_norm


def build_vocab(qa_pairs: List[Dict], top_k: int = 5000, min_freq: int = 2, use_canon: bool = True):
    """
    qa_pairs: list of dicts with keys 'question' and 'answers' (list[str]).
    Returns: (vocab_list, counter) where vocab_list is sorted by descending frequency.
    """
    counter = Counter()
    for ex in qa_pairs:
        q_norm = vqa_normalize(ex["question"])
        for a in ex["answers"]:
            a_norm = vqa_normalize(a)
            a_can  = canonicalize(a_norm, q_norm) if use_canon else a_norm
            if a_can:
                counter[a_can] += 1
    # apply min_freq
    items = [(ans, cnt) for ans, cnt in counter.items() if cnt >= min_freq]
    # sort by frequency desc, then lexicographically for stability
    items.sort(key=lambda x: (-x[3], x))
    # cut by top_k
    vocab = [ans for ans, _ in items[:top_k]]
    return vocab, counter


def save_vocab_json(vocab: List[str], out_path: str):
    """
    Save the vocabulary to a JSON file under the schema {"answers": [...]}.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"answers": vocab}, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab with {len(vocab)} entries to {out_path}")


def load_vqav2_pairs(root: str, splits=("train","val")) -> List[Dict]:
    """
    Load VQAv2 questions+annotations and build qa_pairs:
      root/questions/v2_OpenEnded_mscoco_{split}2014_questions.json
      root/annotations/v2_mscoco_{split}2014_annotations.json
    Returns a list of dicts: {question: str, answers: List[str]}.
    """
    all_pairs = []
    for split in splits:
        qfile = {
            "train": "v2_OpenEnded_mscoco_train2014_questions.json",
            "val":   "v2_OpenEnded_mscoco_val2014_questions.json",
        }[split]
        afile = {
            "train": "v2_mscoco_train2014_annotations.json",
            "val":   "v2_mscoco_val2014_annotations.json",
        }[split]
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
    # Paths and hyperparameters
    VQAv2_ROOT = "/path/to/VQAv2"
    OUT_JSON   = "/path/to/vocab/vqa_vocab_top5k.json"
    TOP_K      = 5000
    MIN_FREQ   = 2
    USE_CANON  = True

    qa_pairs = load_vqav2_pairs(VQAv2_ROOT, splits=("train","val"))  # you may use only "train" if preferred
    vocab, freqs = build_vocab(qa_pairs, top_k=TOP_K, min_freq=MIN_FREQ, use_canon=USE_CANON)
    save_vocab_json(vocab, OUT_JSON)

    # Basic log
    print("Top 20:", vocab[:20])
