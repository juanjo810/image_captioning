from __future__ import annotations


def build_workflow_a_prompt() -> str:
    return """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY these fields. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
Entity ids must be e1, e2, e3... only.
Every entity must include: id, label, category, count_estimate, confidence.
Only include clearly visible entities and visually supported interactions.
Use unknown when uncertain.

Allowed categories: human, animal, object, structure, vehicle, vegetation, tool, food, other.
Allowed indoor_outdoor: indoor, outdoor, mixed, unknown.
Allowed crowd_level: empty, sparse, moderate, dense, unknown.
Allowed activity_level: low, medium, high, unknown.
Allowed lighting: bright, moderate, dim, unknown.
Allowed spatial relations: left_of, right_of, above, below, overlapping.

Return exactly this schema:
{
  "caption": "one concise caption",
  "core": {
    "image_id": "unknown",
    "scene": {"label": "unknown scene", "indoor_outdoor": "unknown", "confidence": 0.5},
    "entities": [
      {"id": "e1", "label": "person", "category": "human", "count_estimate": 1, "confidence": 0.8}
    ],
    "observed_interactions": [
      {"subject_id": "e1", "verb": "ride", "object_id": "e2", "confidence": 0.8}
    ],
    "spatial_relations": [
      {"subject_id": "e1", "relation": "left_of", "object_id": "e2", "confidence": 0.7}
    ],
    "environment": {"crowd_level": "unknown", "activity_level": "unknown", "lighting": "unknown"},
    "caption": "same concise caption"
  }
}

If there are no interactions or spatial relations, use empty arrays.
""".strip()
