#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choice_labels(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


def normalize_answer_index(answer: Any, choices: Optional[List[str]] = None) -> Optional[int]:
    if not choices:
        return None
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < len(choices) else None
    s = str(answer).strip()
    if len(s) == 1 and s.upper().isalpha():
        idx = ord(s.upper()) - ord("A")
        if choices is None or 0 <= idx < len(choices):
            return idx
    if s.isdigit():
        idx = int(s)
        if 0 <= idx < len(choices):
            return idx
    if choices:
        for i, c in enumerate(choices):
            if s.lower() == str(c).strip().lower():
                return i
    return None


def normalize_row(dataset: str, idx: int, question: str, answer: Any,
                  choices: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  source_id: Optional[str] = None) -> Dict[str, Any]:
    choices = [safe_str(c) for c in choices] if choices else None
    answer_index = normalize_answer_index(answer, choices)
    answer_value = choice_labels(len(choices))[answer_index] if answer_index is not None and choices else safe_str(answer)
    return {
        "dataset": dataset,
        "id": source_id or f"{dataset}_{idx}",
        "question": safe_str(question).strip(),
        "choices": choices,
        "answer": answer_value,
        "answer_index": answer_index,
        "metadata": metadata or {},
    }


def _load(name: str, *args: Any, **kwargs: Any) -> Any:
    try:
        return load_dataset(name, *args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {name}: {exc}") from exc


def _limit(rows: List[Dict[str, Any]], max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if max_examples is None or len(rows) <= max_examples:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_examples)


def load_mmlu_pro(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("TIGER-Lab/MMLU-Pro", split="test")
    rows = [normalize_row("mmlu_pro", i, ex.get("question", ""), ex.get("answer"),
                          list(ex.get("options") or ex.get("choices") or []),
                          {"category": ex.get("category"), "src": "TIGER-Lab/MMLU-Pro"},
                          safe_str(ex.get("question_id") or ex.get("id") or f"mmlu_pro_{i}"))
            for i, ex in enumerate(tqdm(ds, desc="MMLU-Pro"))]
    return _limit(rows, max_examples, seed)


def load_gsm8k(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="GSM8K")):
        ans = safe_str(ex.get("answer", ""))
        match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", ans)
        rows.append(normalize_row("gsm8k", i, ex.get("question", ""), match.group(1) if match else ans,
                                  None, {"full_solution": ans, "src": "openai/gsm8k"}))
    return _limit(rows, max_examples, seed)


def load_math500(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("HuggingFaceH4/MATH-500", split="test")
    rows = [normalize_row("math500", i, ex.get("problem", ""), ex.get("answer", ""), None,
                          {"solution": ex.get("solution"), "subject": ex.get("subject"),
                           "level": ex.get("level"), "src": "HuggingFaceH4/MATH-500"})
            for i, ex in enumerate(tqdm(ds, desc="MATH-500"))]
    return _limit(rows, max_examples, seed)


def load_truthfulqa_mc(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("EleutherAI/truthful_qa_mc", split="validation")
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="TruthfulQA-MC")):
        target = ex.get("mc1_targets") if isinstance(ex.get("mc1_targets"), dict) else {}
        choices = target.get("choices") or ex.get("choices")
        labels = target.get("labels")
        answer_idx = next((j for j, lab in enumerate(labels or []) if lab == 1), None)
        if answer_idx is None:
            answer_idx = normalize_answer_index(ex.get("label"), choices)
        if answer_idx is None:
            raise RuntimeError(f"TruthfulQA row {i} has no answer label")
        rows.append(normalize_row("truthfulqa_mc", i, ex.get("question", ""), answer_idx, choices,
                                  {"src": "EleutherAI/truthful_qa_mc"}))
    return _limit(rows, max_examples, seed)


def load_strategyqa(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    try:
        ds = load_dataset("ChilleD/StrategyQA", split="test")
        src = "ChilleD/StrategyQA"
    except Exception:
        ds = _load("voidful/StrategyQA", split="train")
        src = "voidful/StrategyQA"
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="StrategyQA")):
        answer = ex.get("answer")
        ans_idx = 1 if str(answer).lower() == "true" else 0
        rows.append(normalize_row("strategyqa", i, ex.get("question", ""), ans_idx, ["False", "True"],
                                  {"facts": ex.get("facts"), "decomposition": ex.get("decomposition"), "src": src}))
    return _limit(rows, max_examples, seed)


def load_medqa(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("GBaker/MedQA-USMLE-4-options", split="test")
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="MedQA")):
        opts = ex.get("options")
        if isinstance(opts, dict):
            keys = sorted(opts.keys())
            choices = [opts[k] for k in keys]
            answer = ex.get("answer_idx") or ex.get("answer")
            answer_idx = keys.index(answer) if answer in keys else normalize_answer_index(answer, choices)
        else:
            choices = list(opts or [])
            answer_idx = normalize_answer_index(ex.get("answer_idx") or ex.get("answer"), choices)
        rows.append(normalize_row("medqa", i, ex.get("question", ""), answer_idx, choices,
                                  {"src": "GBaker/MedQA-USMLE-4-options"}))
    return _limit(rows, max_examples, seed)


def load_medmcqa(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("openlifescienceai/medmcqa", split="validation")
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="MedMCQA")):
        choices = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
        rows.append(normalize_row("medmcqa", i, ex.get("question", ""), normalize_answer_index(ex.get("cop"), choices),
                                  choices, {"subject_name": ex.get("subject_name"), "topic_name": ex.get("topic_name"),
                                            "exp": ex.get("exp"), "src": "openlifescienceai/medmcqa"}))
    return _limit(rows, max_examples, seed)


def load_aime2024(max_examples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    ds = _load("Maxwell-Jia/AIME_2024", split="train")
    rows = [normalize_row("aime2024", i, ex.get("Problem") or ex.get("problem") or ex.get("question") or "",
                          ex.get("Answer") or ex.get("answer") or "", None,
                          {"src": "Maxwell-Jia/AIME_2024"})
            for i, ex in enumerate(tqdm(ds, desc="AIME_2024"))]
    return _limit(rows, max_examples, seed)


LOADERS = {
    "mmlu_pro": load_mmlu_pro,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "truthfulqa_mc": load_truthfulqa_mc,
    "strategyqa": load_strategyqa,
    "medqa": load_medqa,
    "medmcqa": load_medmcqa,
    "aime2024": load_aime2024,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS.keys()))
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    all_rows = []
    for name in args.datasets:
        if name not in LOADERS:
            raise ValueError(f"Unknown dataset: {name}. Available: {sorted(LOADERS)}")
        rows = LOADERS[name](args.max_examples, args.seed)
        if not rows:
            raise RuntimeError(f"Dataset {name} produced zero rows")
        write_jsonl(rows, out_dir / f"{name}.jsonl")
        all_rows.extend(rows)
        print(f"Wrote {len(rows)} rows for {name}")

    write_jsonl(all_rows, out_dir / "all.jsonl")
    print(f"Wrote combined file with {len(all_rows)} rows")


if __name__ == "__main__":
    main()
