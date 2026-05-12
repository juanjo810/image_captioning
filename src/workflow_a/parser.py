from __future__ import annotations

import json
import re


def extract_json_block(text: str) -> dict:
    text = text.strip()

    fenced = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL)
    candidates = fenced if fenced else [text]

    for candidate in candidates:
        candidate = candidate.strip()

        start = candidate.find("{")
        end = candidate.rfind("}")

        if start == -1 or end == -1:
            continue

        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON object found in model output.")
