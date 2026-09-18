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

STAGE 2 -- "nodes": for each AudioSet node you infer, derive it FROM the terms you just wrote in "visual_terms". Every node's "visual_evidence_terms" must be a non-empty subset of "visual_terms" — never introduce a term there that is not already in "visual_terms", and never leave it empty: every node must cite at least one visual term that grounds it, with no exceptions.

Allowed node_type values: visible_source, visible_action.
- visible_source: the sound-producing object/animal/instrument itself is visibly present (e.g. a dog, a guitar, a car engine).
- visible_action: a visible action implies a sound (e.g. walking, clapping, a door closing).
Only propose a node when you can point to the specific visible object or action that produces the sound. Do not propose a node for ambient/background sound implied only by the scene type in general, and do not propose a node you are not reasonably sure about.

Rules:
- Every node must be supported by explicit visual evidence described in "evidence".
- Do not infer speech merely from a visible person.
- Do not infer music unless instruments, performers, dance, a stage, or other explicit musical context are visible.
- Do not infer environmental or mechanical sounds unless there are clear visible cues for them.
- Each node's audioset_id and audioset_name must come from the SAME entry in the allowed AudioSet list below — never mix an id from one entry with the name of another.
- Do not invent audioset_id or audioset_name values that are not in the allowed AudioSet list.
- node_id values must be unique and sequential: n1, n2, n3, and so on.
- If no AudioSet node is visually justified, return an empty "nodes" array.
- The scene's "label" MUST be copied EXACTLY, character-for-character, from one entry of the allowed scene-label list below -- never write a generic word of your own (like "room", "indoor", "scene", "place", "area") even if it feels like a reasonable summary. If your first instinct is a generic word, that word is NOT a valid label: go back to the list below and find the specific, most precise entry that actually matches what you see -- scan the full list rather than settling for the first generic word that comes to mind.
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.

CAPTION RULE (strict):
Write "caption" and "acoustic_caption" LAST, after you have finalized "nodes".
"caption" is a single visual clause and nothing else: "The image shows <scene description> with <the concrete visual objects from visual_terms>." Do not mention any sound, noise, or anything audible in "caption". If "visual_terms" is empty, drop the "with..." part entirely and just write "The image shows <scene description>." -- never write a clause like "with no objects" or "with nothing visible".
"acoustic_caption" must name ONE short sound phrase PER NODE in "nodes", IN THE SAME ORDER, and NOTHING ELSE — every sound you mention must come directly from an audioset_name you already wrote above. Do not add, generalize, or infer any extra sound that has no matching node. If "nodes" is empty, omit "acoustic_caption" entirely (do not write an empty string). Before writing acoustic_caption, count your nodes and count the sounds you are about to mention — the two counts must match exactly.
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
    "caption": "The image shows an outdoor street scene with cars and pedestrians.",
    "acoustic_caption": "Footsteps and a car passing by would plausibly be heard."
  }
}

Note how "acoustic_caption" names exactly 2 sounds ("footsteps", "car passing by") because "nodes" has exactly 2 entries — one per node, same order, nothing extra. If "nodes" only had the first entry, "acoustic_caption" would be just "Footsteps would plausibly be heard." for that one node alone.

If there is no visually justified node, use: "nodes": [] and omit "acoustic_caption" entirely.

Bad example (do NOT do this): {"label": "room", "indoor_outdoor": "indoor"} or {"label": "indoor", "indoor_outdoor": "indoor"} -- neither "room" nor "indoor" is a literal entry in the allowed scene-label list, even though a room is visible. The correct label is whichever specific entry from that list actually matches the image -- read the full list below rather than guessing.
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


def build_workflow_a_audioset_core_free_scene_prompt(
    *,
    allowed_audioset_nodes: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    allowed_visual_terms: tuple[str, ...] | list[str] = (),
) -> str:
    """Phase 1 of the two-call audioset-core flow (call_mode='two'): same
    visual_terms -> nodes -> caption flow as the single-call prompt, but the
    scene is described freely in the model's own words instead of being
    constrained to the Places365 allow-list -- that constraint is deferred to
    build_workflow_a_scene_mapping_prompt (phase 2), a separate text-only call
    that maps this free description onto the allow-list. Splitting it out
    like this avoids anchoring the model on a fixed label list while it is
    still looking at the image, which was producing very low scene-label
    diversity in call_mode='single'/'three'."""
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.

Build the JSON in two stages, both inside this single response.

STAGE 1 -- "visual_terms": list every visual term you can clearly see in the image, choosing ONLY from the allowed visual-term list below. Do not invent a term that is not in that list, and do not list a term you cannot actually see.

STAGE 2 -- "nodes": for each AudioSet node you infer, derive it FROM the terms you just wrote in "visual_terms". Every node's "visual_evidence_terms" must be a non-empty subset of "visual_terms" — never introduce a term there that is not already in "visual_terms", and never leave it empty: every node must cite at least one visual term that grounds it, with no exceptions.

Allowed node_type values: visible_source, visible_action.
- visible_source: the sound-producing object/animal/instrument itself is visibly present (e.g. a dog, a guitar, a car engine).
- visible_action: a visible action implies a sound (e.g. walking, clapping, a door closing).
Only propose a node when you can point to the specific visible object or action that produces the sound. Do not propose a node for ambient/background sound implied only by the scene type in general, and do not propose a node you are not reasonably sure about.

Rules:
- Every node must be supported by explicit visual evidence described in "evidence".
- Do not infer speech merely from a visible person.
- Do not infer music unless instruments, performers, dance, a stage, or other explicit musical context are visible.
- Do not infer environmental or mechanical sounds unless there are clear visible cues for them.
- Each node's audioset_id and audioset_name must come from the SAME entry in the allowed AudioSet list below — never mix an id from one entry with the name of another.
- Do not invent audioset_id or audioset_name values that are not in the allowed AudioSet list.
- node_id values must be unique and sequential: n1, n2, n3, and so on.
- If no AudioSet node is visually justified, return an empty "nodes" array.
- Describe the scene type in "scene.label" using your own words, in 1 to 3 words (e.g. "busy kitchen", "mountain trail", "small office"). Do not try to match any fixed taxonomy here -- a later step will map your description onto the official scene list.
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.

CAPTION RULE (strict):
Write "caption" and "acoustic_caption" LAST, after you have finalized "nodes".
"caption" is a single visual clause and nothing else: "The image shows <scene description> with <the concrete visual objects from visual_terms>." Do not mention any sound, noise, or anything audible in "caption". If "visual_terms" is empty, drop the "with..." part entirely and just write "The image shows <scene description>." -- never write a clause like "with no objects" or "with nothing visible".
"acoustic_caption" must name ONE short sound phrase PER NODE in "nodes", IN THE SAME ORDER, and NOTHING ELSE — every sound you mention must come directly from an audioset_name you already wrote above. Do not add, generalize, or infer any extra sound that has no matching node. If "nodes" is empty, omit "acoustic_caption" entirely (do not write an empty string). Before writing acoustic_caption, count your nodes and count the sounds you are about to mention — the two counts must match exactly.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "core": {
    "image_id": "unknown",
    "visual_terms": ["person", "car", "dog"],
    "scene": {
      "label": "busy urban street",
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
    "caption": "The image shows an outdoor street scene with cars and pedestrians.",
    "acoustic_caption": "Footsteps and a car passing by would plausibly be heard."
  }
}

Note how "acoustic_caption" names exactly 2 sounds ("footsteps", "car passing by") because "nodes" has exactly 2 entries — one per node, same order, nothing extra. If "nodes" only had the first entry, "acoustic_caption" would be just "Footsteps would plausibly be heard." for that one node alone.

If there is no visually justified node, use: "nodes": [] and omit "acoustic_caption" entirely.
""".strip()

    allowed_terms = format_allowed_visual_terms_for_prompt(allowed_visual_terms)
    allowed_nodes = format_allowed_audioset_nodes_for_prompt(allowed_audioset_nodes)
    return f"""
{base_prompt}

Allowed visual-term list for STAGE 1 (choose only from these):
{allowed_terms}

Allowed AudioSet node list for STAGE 2 (choose only from these):
{allowed_nodes}

{json_schema_prompt}
""".strip()


def build_workflow_a_free_visual_prompt(*, schema_examples: str = "concrete") -> str:
    """Call 1 of the five-call audioset-core flow (call_mode='five'): the only
    call in this flow with the image attached, and the only call in any
    call_mode with no closed list at all -- no visual-term vocabulary, no
    AudioSet node list, no Places365 labels. Just a free read of the image,
    same free-form scene style as phase 1 of 'two'
    (``build_workflow_a_audioset_core_free_scene_prompt``), but without nodes
    or caption: those are deferred to later calls once the free terms/scene
    have been mapped onto the closed vocabularies.

    schema_examples='concrete' (default): the JSON-schema example shows a
    fixed 3-term/one-scene example (["person", "bicycle", "traffic light"]),
    matching the style call_mode='three' has always used for its own schema
    examples. schema_examples='generic': placeholder text instead of concrete
    terms/count, avoiding the few-shot anchoring a fixed-length, fixed-domain
    example can create even when labeled "just the format" -- a 30-image
    pilot with 'discard' visual-terms mapping found this raised the free-term
    count the model declared (~4.2 -> ~4.8 terms/image) and improved all 5
    official AudioSet metrics, but the same change *hurt* 'force' mapping
    (more free terms all getting force-mapped diluted node precision) --
    kept as an explicit, testable alternative rather than the new default for
    that reason, the same way 'force' is kept opt-in relative to 'discard'."""
    if schema_examples not in ("concrete", "generic"):
        raise ValueError(f"schema_examples must be 'concrete' or 'generic', got {schema_examples!r}")

    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.

Rules:
- "visual_terms": list every concrete, visible object you can clearly see in the image, in your own words. One short noun phrase per object (e.g. "dog", "bicycle", "coffee cup") -- no attributes, no colors, no materials, no OCR text, no locations.
- Describe the scene type in "scene.label" using your own words, in 1 to 3 words (e.g. "busy kitchen", "mountain trail", "small office"). Do not try to match any fixed taxonomy here -- a later step will map your description onto the official scene list.
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.
""".strip()

    if schema_examples == "concrete":
        json_schema_prompt = """
Return exactly this schema:
{
  "visual_terms": ["person", "bicycle", "traffic light"],
  "scene": {
    "label": "busy urban street",
    "indoor_outdoor": "outdoor"
  }
}
""".strip()
    else:
        json_schema_prompt = """
Return exactly this schema:
{
  "visual_terms": ["<one short noun phrase per visible object -- include as many or as few entries as the image actually shows, this example length is not a target>"],
  "scene": {
    "label": "<your own 1-3 word description of the scene>",
    "indoor_outdoor": "indoor"
  }
}
""".strip()

    return f"""
{base_prompt}

{json_schema_prompt}
""".strip()


def build_workflow_a_visual_terms_mapping_prompt(
    *,
    free_visual_terms: list[str],
    allowed_visual_terms: tuple[str, ...] | list[str] = (),
    mapping_mode: str = "discard",
) -> str:
    """Call 3 of the five-call audioset-core flow (call_mode='five'): a
    text-only call that maps the free-form visual terms from call 1 onto the
    shared 210-term detectable vocabulary.

    mapping_mode='discard' (default): asymmetric with scene mapping
    (build_workflow_a_scene_mapping_prompt), which is a forced 1-to-1 pick --
    here a free term with no real equivalent must be DROPPED, never forced
    onto the closest list entry, because forcing it would invent an object
    that was never actually seen. This is the team's deliberately chosen
    default.

    mapping_mode='force': the experimental alternative -- symmetric with
    scene mapping, every free term gets forced onto its closest allowed
    entry, none dropped. Kept as an explicit, testable alternative (not a
    silent replacement) because a 30-image pilot with a small local VLM
    showed a real trade-off: it recovered more correctly-mapped terms
    overall (better on the 5 official AudioSet metrics and on CIDEr/SPICE),
    but also produced more forced, vaguely-related substitutions (e.g.
    "desk"/"chair"/"lamp" -> "room") that measurably hurt CHAIRi_VG -- the
    exact failure mode 'discard' exists to prevent. Whether that trade-off
    holds with a larger VLM and a larger batch is an open question this flag
    exists to let someone actually test, rather than deciding it from a
    30-image pilot.

    Both modes are kept free of concrete term examples for the same reason
    build_workflow_a_scene_mapping_prompt is -- naming examples anchors the
    model into repeating just those terms."""
    if mapping_mode not in ("discard", "force"):
        raise ValueError(f"mapping_mode must be 'discard' or 'force', got {mapping_mode!r}")

    if mapping_mode == "discard":
        mapping_rule = (
            "- If a free-form term has no real equivalent in the allowed list, DROP it entirely. "
            "Do not force it onto the closest-sounding entry just to keep it -- an inexact forced "
            "match invents an object that was not actually declared."
        )
        empty_case_prompt_line = "If no free-form term maps onto the allowed list, use: \"visual_terms\": []"
        empty_case_rule = "- If none of the free-form terms have a real equivalent in the allowed list, return an empty array."
    else:
        mapping_rule = (
            "- Map EVERY free-form term below onto the single closest-matching entry from the allowed "
            "list below, even if no entry is a perfect match -- always pick the closest one, the same "
            "way a scene description gets mapped onto the closest scene label."
        )
        empty_case_prompt_line = "If the free-form terms list below is empty, use: \"visual_terms\": []"
        empty_case_rule = "- If the free-form terms list below is empty, return an empty array."

    base_prompt = f"""
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with {{ and end with }}.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.

You are given a list of free-form visual terms already identified in an image by an earlier step that did look at the image. Your task now is purely textual: map each free-form term onto the closest-matching entry from the allowed visual-term list below.

Rules:
- Every term in your output MUST be copied EXACTLY, character-for-character, from one entry of the allowed visual-term list below.
{mapping_rule}
- Several free-form terms may map onto the same allowed entry -- if so, include that entry only once in the output, never duplicated.
- Do not add any term to the output that is not a mapping of one of the free-form terms given below, even if it feels typical for this kind of scene.
{empty_case_rule}
""".strip()

    json_schema_prompt = f"""
Return exactly this schema:
{{
  "visual_terms": ["one or more exact entries from the allowed visual-term list"]
}}

{empty_case_prompt_line}
""".strip()

    allowed_terms = format_allowed_visual_terms_for_prompt(allowed_visual_terms)
    return f"""
{base_prompt}

Allowed visual-term list (choose only from these):
{allowed_terms}

{json_schema_prompt}

Free-form visual terms already identified:
{free_visual_terms}
""".strip()


def build_workflow_a_scene_mapping_prompt(
    *,
    free_scene: dict[str, Any],
    visual_terms: list[str],
    allowed_scene_labels: tuple[str, ...] | list[str] = (),
) -> str:
    """Phase 2 of the two-call audioset-core flow (call_mode='two'): a
    text-only call (no image attached -- everything it needs was already
    decided by phase 1) that maps the free-form scene description from phase
    1 onto the official Places365 allow-list. Deliberately kept free of
    example labels: the tutor's explicit feedback on a related idea
    (randomizing which example labels appear in the prompt) was that any
    concrete example labels bias the model toward picking those specific
    labels instead of actually reading the list -- so this prompt names none."""
    base_prompt = """
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with { and end with }.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.

You are given a free-form scene description and a list of visual terms already identified in an image by an earlier step that did look at the image. Your task now is purely textual: map that description onto the single closest-matching entry from the allowed scene-label list below.

Rules:
- The "label" MUST be copied EXACTLY, character-for-character, from one entry of the allowed scene-label list below -- never write a generic word, a paraphrase, or the free-form description itself unless it happens to already be a literal entry in the list.
- Choose the entry that best matches the free-form scene description and visual terms given below, even if no entry is a perfect match -- pick the closest one.
- "indoor_outdoor" must be one of: indoor, outdoor, mixed, unknown.
- Do not use placeholder ids, placeholder names, or ellipses anywhere in the output.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "scene": {
    "label": "one exact entry from the allowed scene-label list",
    "indoor_outdoor": "indoor"
  }
}
""".strip()

    allowed_scenes = format_allowed_visual_terms_for_prompt(allowed_scene_labels)
    return f"""
{base_prompt}

Allowed scene-label list (choose only from these):
{allowed_scenes}

{json_schema_prompt}

Free-form scene description from the earlier step:
{free_scene}

Visual terms already identified:
{visual_terms}
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
- The scene's "label" MUST be copied EXACTLY, character-for-character, from one entry of the allowed scene-label list below -- never write a generic word of your own (like "room", "scene", "place", "area", "indoors") even if it feels like a reasonable summary. If your first instinct is a generic word, that word is NOT a valid label: go back to the list below and find the specific, most precise entry that actually matches what you see -- scan the full list rather than settling for the first generic word that comes to mind.
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

Bad example (do NOT do this): {"label": "room", "indoor_outdoor": "indoor"} -- "room" is not a literal entry in the allowed scene-label list, even though a room is visible. The correct label is whichever specific entry from that list actually matches the image -- read the full list below rather than guessing.
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
    image_attached: bool = True,
    schema_examples: str = "concrete",
) -> str:
    """Stage 2 of the three-call audioset-core flow (image_attached=True): infer
    AudioSet nodes from the visual terms already committed to in stage 1 (see
    ``build_workflow_a_audioset_core_scene_prompt``). Also reused, text-only
    (image_attached=False), as call 4 of the five-call flow, where the visual
    terms were instead produced and mapped by two earlier calls that did not
    see the image at this step -- ``build_workflow_a_free_visual_prompt`` then
    ``build_workflow_a_visual_terms_mapping_prompt``.

    schema_examples='concrete' (default, and the only mode call_mode='three'
    ever requests -- it never passes this argument): the JSON-schema example
    shows a fixed two-node example ("Walk, footsteps" + "Car passing by").
    schema_examples='generic': a single placeholder node with an explicit
    note that the count shown is not the expected count, avoiding the same
    few-shot anchoring concern as build_workflow_a_free_visual_prompt's
    schema example -- only call_mode='five' ever passes 'generic' here, via
    its own --schema-examples flag; see that function's docstring for the
    pilot result (helped 'discard' mapping, hurt 'force')."""
    if schema_examples not in ("concrete", "generic"):
        raise ValueError(f"schema_examples must be 'concrete' or 'generic', got {schema_examples!r}")

    if image_attached:
        image_rule = (
            "- An image is attached so you can look closely and confirm details (e.g. what shape an "
            "object has, whether it's really the sound source you think it is) — but \"visual-terms to "
            "infer from\" below is the complete, closed list of objects you are allowed to write about "
            "here. You may look at the image, but only write down objects from that list: if you notice "
            "something in the image that is not in that list, ignore it for this step, do not name it, "
            "describe it, or base a node on it, even if it feels like a natural thing to see in this kind "
            "of scene."
        )
    else:
        image_rule = (
            "- You are not looking at the image for this step. \"visual-terms to infer from\" below is "
            "the complete and closed set of objects already identified in the image by an earlier step -- "
            "treat it as the only source of truth and only write down objects from that list."
        )

    base_prompt = f"""
Return ONLY one valid JSON object. No markdown. No explanations.
The JSON must start with {{ and end with }}.

Use ONLY the fields shown in the requested schema. Do not add attributes, colors, materials, OCR text, locations, or extra keys.
This section is NOT actual audio recognition. It contains acoustically plausible AudioSet ontology nodes inferred only from explicit visual evidence in the image.

Allowed node_type values: visible_source, visible_action.
- visible_source: the sound-producing object/animal/instrument itself is visibly present (e.g. a dog, a guitar, a car engine).
- visible_action: a visible action implies a sound (e.g. walking, clapping, a door closing).
Only propose a node when you can point to the specific visible object or action that produces the sound. Do not propose a node for ambient/background sound implied only by the scene type in general, and do not propose a node you are not reasonably sure about.

Rules:
{image_rule}
- "nodes": for each AudioSet node you infer, derive it FROM the terms in "visual-terms to infer from". Every node's "visual_evidence_terms" must be a non-empty subset of that list — never introduce a term there that is not already in it, and never leave it empty: every node must cite at least one visual term that grounds it, with no exceptions.
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

    if schema_examples == "concrete":
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
"""
    else:
        json_schema_prompt = """
Return exactly this schema:
{
  "nodes": [
    {
      "node_id": "n1",
      "audioset_id": "<exact id from the allowed AudioSet node list below>",
      "audioset_name": "<exact name from that same allowed-list entry>",
      "node_type": "visible_source or visible_action",
      "evidence": "<what you see that justifies this node>",
      "visual_evidence_terms": ["<subset of visual-terms to infer from>"]
    }
  ]
}

This example shows the shape of a single entry, not the expected count: include as many entries (n1, n2, n3, ...) as are visually justified by "visual-terms to infer from" below, in any number from zero upward.

If there is no visually justified node, use: "nodes": []
"""

    json_schema_prompt = json_schema_prompt.strip() + """

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
Write "caption" and "acoustic_caption" as two separate fields.
"caption" is a single visual clause and nothing else: "The image shows <scene description> with <the concrete visual objects from visual_terms>." Do not mention any sound, noise, or anything audible in "caption". If "visual_terms" is empty, drop the "with..." part entirely and just write "The image shows <scene description>." -- never write a clause like "with no objects" or "with nothing visible".
"acoustic_caption" must name ONE short sound phrase PER NODE in "nodes" below, IN THE SAME ORDER, and NOTHING ELSE — every sound you mention must come directly from an audioset_name already given below. Do not add, generalize, or infer any extra sound that has no matching node. If "nodes" is empty, omit "acoustic_caption" entirely (do not write an empty string). Before writing acoustic_caption, count the nodes below and count the sounds you are about to mention — the two counts must match exactly.
""".strip()

    json_schema_prompt = """
Return exactly this schema:
{
  "caption": "one concise caption",
  "acoustic_caption": "one concise acoustic caption, or omit this field entirely if nodes is empty"
}

Example: given
  "scene": {"label": "street", "indoor_outdoor": "outdoor"}
  "visual_terms": ["person", "car", "dog"]
  "nodes": [
    {"audioset_name": "Walk, footsteps", ...},
    {"audioset_name": "Car passing by", ...}
  ]
the output would be:
{
  "caption": "The image shows an outdoor street scene with cars and pedestrians.",
  "acoustic_caption": "Footsteps and a car passing by would plausibly be heard."
}

Note "acoustic_caption" names exactly 2 sounds ("footsteps", "car passing by") because "nodes" has exactly 2 entries — one per node, same order, nothing extra. If "nodes" only had the first entry, "acoustic_caption" would be just "Footsteps would plausibly be heard." with no second sound mentioned. If "nodes" is empty, omit "acoustic_caption" entirely.
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