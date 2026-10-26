import json
import os
from typing import List, Dict, Any, Set

class BenchmarkLoader:
    @staticmethod
    def load_jsonl(file_path: str, limit: int = 0) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        items = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
                if 0 < limit <= len(items):
                    break
        return items

    @staticmethod
    def check_contamination(item_text: str, training_corpus_tokens: Set[str], n: int = 8) -> float:
        words = item_text.lower().split()
        if len(words) < n:
            return 0.0
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        matches = sum(1 for ng in ngrams if ng in training_corpus_tokens)
        return matches / len(ngrams) if ngrams else 0.0
