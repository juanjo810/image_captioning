from __future__ import annotations

from src.workflow_a.audioset_nodes import (
    format_allowed_audioset_nodes_for_prompt,
    format_allowed_visual_terms_for_prompt,
)


def build_workflow_a_legacy_visual_prompt(
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


def build_workflow_a_audioset_core_prompt(
    *,
    allowed_audioset_nodes: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    allowed_visual_terms: tuple[str, ...] | list[str] = (),
    allowed_scene_labels: tuple[str, ...] | list[str] = (),
) -> str:
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.

Build the JSON in two stages, both inside this single response.

STAGE 1 -- "visual_terms": list every visual term you can clearly see in the image, choosing ONLY from the allowed visual-term list below. Do not invent a term that is not in that list, and do not list a term you cannot actually see.

STAGE 2 -- "nodes": for each AudioSet node you infer, derive it FROM the terms you just wrote in "visual_terms". Every node's "visual_evidence_terms" must be a subset of "visual_terms" — never introduce a term there that is not already in "visual_terms". The only exception is a "scene_affordance" node (ambient sound implied by the scene type itself, not by one specific visible object): for that node type you may omit "visual_evidence_terms" entirely.

Allowed node_type values: visible_source, visible_action, scene_affordance, uncertain.
- visible_source: the sound-producing object/animal/instrument itself is visibly present (e.g. a dog, a guitar, a car engine).
- visible_action: a visible action implies a sound (e.g. walking, clapping, a door closing).
- scene_affordance: the scene type plausibly implies ambient sound even without a specific visible source (e.g. an urban street implies traffic ambience).
- uncertain: visual evidence suggests a possible sound but is not conclusive.

Rules:
- Every node must be supported by explicit visual evidence described in "evidence".
- Do not infer speech merely from a visible person.
- Do not infer music unless instruments, performers, dance, a stage, or other explicit musical context are visible.
- Do not infer environmental or mechanical sounds unless there are clear visible cues for them.
- Each node's audioset_id and audioset_name must come from the SAME entry in the allowed AudioSet list below — never mix an id from one entry with the name of another.
- Do not invent audioset_id or audioset_name values that are not in the allowed AudioSet list.
- node_id values must be unique and sequential: n1, n2, n3, and so on.
- If no AudioSet node is visually justified, return an empty "nodes" array.
- The scene's "label" must be chosen from the allowed scene-label list below, as the closest description of the overall scene.
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.

CAPTION RULE (strict):
Write "caption" LAST, after you have finalized "nodes". Structure it as exactly two clauses:
"The image shows <scene description> with <the concrete visual objects from visual_terms>, where <sounds> would plausibly be heard."
The <sounds> clause must name ONE short sound phrase PER NODE in "nodes", IN THE SAME ORDER, and NOTHING ELSE — every sound you mention must come directly from an audioset_name you already wrote above. Do not add, generalize, or infer any extra sound that has no matching node. If "nodes" is empty, drop the "where..." clause entirely and just describe the visible objects. Before writing the caption, count your nodes and count the sounds you are about to mention — the two counts must match exactly.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "core": {
    "image_id": "unknown",
    "visual_terms": ["person", "car", "dog"],
    "scene": {
      "label": "street",
      "indoor_outdoor": "outdoor"
    },
    "nodes": [
      {
        "node_id": "n1",
        "audioset_id": "/m/07qv_x0",
        "audioset_name": "Walk, footsteps",
        "node_type": "visible_action",
        "evidence": "people walking on the pavement",
        "visual_evidence_terms": ["person"]
      },
      {
        "node_id": "n2",
        "audioset_id": "/t/dd00134",
        "audioset_name": "Car passing by",
        "node_type": "visible_source",
        "evidence": "a car visible on the street",
        "visual_evidence_terms": ["car"]
      }
    ],
    "caption": "The image shows an outdoor street scene with cars and pedestrians, where footsteps and a car passing by would plausibly be heard."
  }
}

Note how the caption's sound clause has exactly 2 sounds ("footsteps", "car sounds") because "nodes" has exactly 2 entries — one per node, same order, nothing extra. If "nodes" only had the first entry, the caption would end at "...pedestrians." with no "where..." clause needed for a single missing sound, or just "...where footsteps would plausibly be heard." for that one node alone.

If there is no visually justified node, use: "nodes": []
""".strip()

    allowed_terms = format_allowed_visual_terms_for_prompt(allowed_visual_terms)
    allowed_nodes = format_allowed_audioset_nodes_for_prompt(allowed_audioset_nodes)
    allowed_scenes = format_allowed_visual_terms_for_prompt(allowed_scene_labels)
    return f"""
{base_prompt}

Allowed visual-term list for STAGE 1 (choose only from these):
{allowed_terms}

Allowed AudioSet node list for STAGE 2 (choose only from these):
{allowed_nodes}

Allowed scene-label list (choose only from these):
{allowed_scenes}

{json_schema_prompt}
""".strip()