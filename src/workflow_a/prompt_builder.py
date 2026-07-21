from __future__ import annotations

from src.workflow_a.audioset_nodes import format_allowed_audioset_nodes_for_prompt


def build_workflow_a_prompt(
    *,
    include_audioset_nodes: bool = False,
    allowed_audioset_nodes: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
) -> str:
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
Entity ids must be e1, e2, e3, and so on only.
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

    if not include_audioset_nodes:
        return base_prompt

    allowed_nodes = format_allowed_audioset_nodes_for_prompt(allowed_audioset_nodes)

    return f"""
{base_prompt}

Also add a top-level sibling section named "acoustic_semantics".
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.
Use ONLY AudioSet nodes from this allowed list:
{allowed_nodes}

Rules for acoustic_semantics:
- Return "acoustic_semantics" with a "nodes" array.
- Each node must include exactly: id, name, evidence, confidence, inference_type.
- inference_type must be one of: visible_source, visible_action, scene_affordance, uncertain.
- Every selected node must be supported by explicit visual evidence.
- Use only leaf nodes from the allowed list when the required visual evidence is explicit.
- Do not infer speech merely from a visible person.
- Do not infer music unless instruments, performers, dance, stage, or musical context are visible.
- Do not infer environmental sounds unless there are clear visible cues.
- If no AudioSet node is visually justified, return an empty nodes array.
- Do not use placeholder ids, placeholder names, or ellipses.

Add this sibling section after "core":
"acoustic_semantics": {{
  "nodes": [
    {{
      "id": "one id from the allowed list",
      "name": "the exact matching name from the allowed list",
      "evidence": "visible evidence in the image",
      "confidence": 0.7,
      "inference_type": "visible_source"
    }}
  ]
}}
""".strip()
