import json
from typing import Any, Dict

def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from raw LLM text without using regex.
    Finds the outermost balanced JSON boundaries and parses via json.loads.
    """
    text = raw_text.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("No valid JSON object boundaries found in LLM output")
    candidate = text[start_idx : end_idx + 1]
    return json.loads(candidate, strict=False)
