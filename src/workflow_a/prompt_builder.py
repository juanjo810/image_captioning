from __future__ import annotations

from typing import Any

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
- The scene's "label" MUST be copied EXACTLY, character-for-character, from one entry of the allowed scene-label list below -- never write a generic word of your own (like "room", "indoor", "scene", "place", "area") even if it feels like a reasonable summary. If your first instinct is a generic word, that word is NOT a valid label: go back to the list and find the specific entry that best matches what you were about to say (e.g. instead of "room" or "indoor", scan the list for the specific kind of room you actually see, such as "office", "computer room", "home office", "bedroom", "classroom", etc.).
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

Bad example (do NOT do this): {"label": "room", "indoor_outdoor": "indoor"} or {"label": "indoor", "indoor_outdoor": "indoor"} -- neither "room" nor "indoor" is a literal entry in the allowed scene-label list, even though a room is visible. The correct label is whichever specific entry from that list (e.g. "office", "computer room", "bedroom") actually matches the image.
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


def build_workflow_a_audioset_core_scene_prompt(
    *,
    allowed_visual_terms: tuple[str, ...] | list[str] = (),
    allowed_scene_labels: tuple[str, ...] | list[str] = (),
) -> str:
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.

Rules:
- "visual_terms": list every visual term you can clearly see in the image, choosing ONLY from the allowed visual-term list below. Do not invent a term that is not in that list, and do not list a term you cannot actually see.
- The scene's "label" MUST be copied EXACTLY, character-for-character, from one entry of the allowed scene-label list below -- never write a generic word of your own (like "room", "scene", "place", "area", "indoors") even if it feels like a reasonable summary. If your first instinct is a generic word, that word is NOT a valid label: go back to the list and find the specific entry that best matches what you were about to say (e.g. instead of "room", scan the list for the specific kind of room you actually see, such as "office", "computer room", "home office", "bedroom", "classroom", etc.).
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "image_id": "unknown",
  "visual_terms": ["person", "car", "dog"],
  "scene": {
    "label": "street",
    "indoor_outdoor": "outdoor"
  }
}

Bad example (do NOT do this): {"label": "room", "indoor_outdoor": "indoor"} -- "room" is not a literal entry in the allowed scene-label list, even though a room is visible. The correct label is whichever specific entry from that list (e.g. "office", "computer room", "bedroom") actually matches the image.
""".strip()

    allowed_terms = format_allowed_visual_terms_for_prompt(allowed_visual_terms)
    allowed_scenes = format_allowed_visual_terms_for_prompt(allowed_scene_labels)
    return f"""
{base_prompt}

Allowed visual-term list (choose only from these):
{allowed_terms}

Allowed scene-label list (choose only from these):
{allowed_scenes}

{json_schema_prompt}
""".strip()


def build_workflow_a_audioset_core_nodes_prompt(
    *,
    visual_terms: list[str],
    allowed_audioset_nodes: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
) -> str:
    """Stage 2 of the three-call audioset-core flow: infer AudioSet nodes from the
    visual terms already committed to in stage 1 (see
    ``build_workflow_a_audioset_core_scene_prompt``)."""
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.

Allowed node_type values: visible_source, visible_action, scene_affordance, uncertain.
- visible_source: the sound-producing object/animal/instrument itself is visibly present (e.g. a dog, a guitar, a car engine).
- visible_action: a visible action implies a sound (e.g. walking, clapping, a door closing).
- scene_affordance: the scene type plausibly implies ambient sound even without a specific visible source (e.g. an urban street implies traffic ambience).
- uncertain: visual evidence suggests a possible sound but is not conclusive.

Rules:
- An image is attached so you can look closely and confirm details (e.g. what shape an object has, whether it's really the sound source you think it is) — but "visual-terms to infer from" below is the complete, closed list of objects you are allowed to write about here. You may look at the image, but only write down objects from that list: if you notice something in the image that is not in that list, ignore it for this step, do not name it, describe it, or base a node on it, even if it feels like a natural thing to see in this kind of scene.
- "nodes": for each AudioSet node you infer, derive it FROM the terms in "visual-terms to infer from". Every node's "visual_evidence_terms" must be a subset of that list — never introduce a term there that is not already in it. The only exception is a "scene_affordance" node (ambient sound implied by the scene type itself, not by one specific visible object): for that node type you may omit "visual_evidence_terms" entirely.
- The "evidence" text itself must also only reference objects from "visual-terms to infer from" — do not mention an object in "evidence" that is not in that list, even in passing.
- Every node must be supported by explicit visual evidence described in "evidence".
- Do not infer speech merely from a visible person.
- Do not infer music unless instruments, performers, dance, a stage, or other explicit musical context are visible.
- Do not infer environmental or mechanical sounds unless there are clear visible cues for them.
- Each node's audioset_id and audioset_name must come from the SAME entry in the allowed AudioSet list below — never mix an id from one entry with the name of another.
- Do not invent audioset_id or audioset_name values that are not in the allowed AudioSet list.
- The chosen audioset_name must be a plausible sound for the visual_evidence_terms cited — do not pair a node with a sound that has no real connection to the object (e.g. do not pair a "keyboard" with "Chainsaw", or a "cup" with "Cowbell", just because both happen to be in the allowed list).
- node_id values must be unique and sequential: n1, n2, n3, and so on.
- If no AudioSet node is visually justified, return an empty "nodes" array.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
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
  ]
}

If there is no visually justified node, use: "nodes": []

Bad example (do NOT do this): if "visual-terms to infer from" is ["person", "car"], do not produce a node like
{"audioset_name": "Cowbell", "evidence": "a cowbell is visible on a pole", "visual_evidence_terms": ["car"]}
-- "pole" and "cowbell" were never declared as visible, and tagging the node with "car" just because it is a known term does not make the evidence true. Only write nodes whose evidence and visual_evidence_terms are built entirely out of ["person", "car"].
""".strip()

    allowed_nodes = format_allowed_audioset_nodes_for_prompt(allowed_audioset_nodes)

    return f"""
{base_prompt}

Allowed AudioSet node list (choose only from these):
{allowed_nodes}

{json_schema_prompt}

visual-terms to infer from (choose only from these):
{visual_terms}
""".strip()


def build_workflow_a_audioset_core_caption_prompt(
    *,
    scene: dict[str, Any],
    visual_terms: list[str],
    nodes: list[dict[str, Any]],
) -> str:
    """Stage 3 of the three-call audioset-core flow: caption built from the
    stage-1 scene/visual_terms and the stage-2 nodes, passed in verbatim
    below so the sound clause stays traceable to nodes already decided."""

    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.

CAPTION RULE (strict):
Structure it as exactly two clauses:
"The image shows <scene description> with <the concrete visual objects from visual_terms>, where <sounds> would plausibly be heard."
The <sounds> clause must name ONE short sound phrase PER NODE in "nodes" below, IN THE SAME ORDER, and NOTHING ELSE — every sound you mention must come directly from an audioset_name already given below. Do not add, generalize, or infer any extra sound that has no matching node. If "nodes" is empty, drop the "where..." clause entirely and just describe the visible objects. Before writing the caption, count the nodes below and count the sounds you are about to mention — the two counts must match exactly.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "caption": "one concise caption"
}

Example: given
  "scene": {"label": "street", "indoor_outdoor": "outdoor"}
  "visual_terms": ["person", "car", "dog"]
  "nodes": [
    {"audioset_name": "Walk, footsteps", ...},
    {"audioset_name": "Car passing by", ...}
  ]
the caption would be:
{
  "caption": "The image shows an outdoor street scene with cars and pedestrians, where footsteps and a car passing by would plausibly be heard."
}

Note the caption's sound clause has exactly 2 sounds ("footsteps", "car passing by") because "nodes" has exactly 2 entries — one per node, same order, nothing extra. If "nodes" only had the first entry, the caption would end "...where footsteps would plausibly be heard." with no second sound mentioned.
""".strip()

    return f"""
{base_prompt}

{json_schema_prompt}

Scene:
{scene}

Visual_terms:
{visual_terms}

Nodes:
{nodes}
""".strip()