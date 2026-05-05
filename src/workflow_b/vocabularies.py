from __future__ import annotations


# ---------------------------------------------------------------------
# Places365 scene groups
# ---------------------------------------------------------------------

RURAL_TRADITIONAL_SCENES = {
    "farm",
    "field cultivated",
    "field wild",
    "field road",
    "hayfield",
    "pasture",
    "orchard",
    "vegetable garden",
    "wheat field",
    "rice paddy",
    "vineyard",
    "village",
    "stable",
    "barn",
    "barndoor",
    "corral",
    "tree farm",
    "yard",
    "cottage",
    "house",
    "cabin outdoor",
    "hunting lodge outdoor",
}

NATURAL_OUTDOOR_SCENES = {
    "forest broadleaf",
    "broadleaf",
    "forest path",
    "forest road",
    "bamboo forest",
    "rainforest",
    "mountain",
    "mountain path",
    "valley",
    "river",
    "creek",
    "lake natural",
    "pond",
    "marsh",
    "swamp",
    "waterfall",
    "watering hole",
    "lawn",
    "park",
}

MARKET_PUBLIC_SCENES = {
    "market outdoor",
    "market indoor",
    "bazaar outdoor",
    "bazaar indoor",
    "flea market indoor",
    "general store outdoor",
    "general store indoor",
    "shopfront",
    "plaza",
    "courtyard",
}

RELIGIOUS_HERITAGE_SCENES = {
    "church outdoor",
    "church indoor",
    "mosque outdoor",
    "synagogue outdoor",
    "temple asia",
    "pagoda",
    "ruin",
    "castle",
    "palace",
    "cemetery",
    "mausoleum",
    "archaelogical excavation",
}

INDOOR_DOMESTIC_SCENES = {
    "kitchen",
    "bedroom",
    "living room",
    "dining room",
    "pantry",
    "attic",
    "basement",
    "home office",
    "shed",
    "storage room",
}

URBAN_TRANSPORT_SCENES = {
    "street",
    "alley",
    "driveway",
    "parking lot",
    "railroad track",
    "train station platform",
    "bridge",
    "road",
    "desert road",
    "highway",
}


SCENE_GROUPS = {
    "rural_traditional": RURAL_TRADITIONAL_SCENES,
    "natural_outdoor": NATURAL_OUTDOOR_SCENES,
    "market_public": MARKET_PUBLIC_SCENES,
    "religious_heritage": RELIGIOUS_HERITAGE_SCENES,
    "indoor_domestic": INDOOR_DOMESTIC_SCENES,
    "urban_transport": URBAN_TRANSPORT_SCENES,
}


OUTDOOR_SCENE_GROUPS = {
    "rural_traditional",
    "natural_outdoor",
    "market_public",
    "religious_heritage",
    "urban_transport",
}

INDOOR_SCENE_GROUPS = {
    "indoor_domestic",
}


def normalize_scene_label(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("/", " ")


def scene_groups_for_label(scene_label: str) -> list[str]:
    normalized = normalize_scene_label(scene_label)
    groups = []

    for group_name, labels in SCENE_GROUPS.items():
        if normalized in labels:
            groups.append(group_name)

    return groups


def infer_indoor_outdoor_from_scene(scene_label: str) -> str:
    groups = set(scene_groups_for_label(scene_label))

    if groups & INDOOR_SCENE_GROUPS:
        return "indoor"

    if groups & OUTDOOR_SCENE_GROUPS:
        return "outdoor"

    normalized = normalize_scene_label(scene_label)

    if "indoor" in normalized:
        return "indoor"

    if "outdoor" in normalized:
        return "outdoor"

    return "unknown"


# ---------------------------------------------------------------------
# Grounding DINO vocabularies
# ---------------------------------------------------------------------

CORE_OBJECT_VOCAB = [
    "person",
    "man",
    "woman",
    "child",
    "horse",
    "donkey",
    "cow",
    "sheep",
    "dog",
    "cat",
    "cart",
    "wagon",
    "basket",
    "tool",
    "tree",
    "plant",
    "house",
    "building",
    "fence",
]

RURAL_OBJECT_VOCAB = [
    "person",
    "man",
    "woman",
    "child",
    "horse",
    "donkey",
    "cow",
    "sheep",
    "goat",
    "dog",
    "cart",
    "wagon",
    "plow",
    "basket",
    "bucket",
    "tool",
    "hay",
    "crop",
    "tree",
    "plant",
    "fence",
    "barn",
    "stable",
    "house",
]

MARKET_OBJECT_VOCAB = [
    "person",
    "man",
    "woman",
    "child",
    "basket",
    "table",
    "stall",
    "cart",
    "wagon",
    "fruit",
    "vegetables",
    "bread",
    "food",
    "cloth",
    "container",
    "bag",
]

RELIGIOUS_HERITAGE_OBJECT_VOCAB = [
    "person",
    "man",
    "woman",
    "child",
    "building",
    "church",
    "temple",
    "stone",
    "arch",
    "door",
    "cross",
    "statue",
    "tree",
]

INDOOR_DOMESTIC_OBJECT_VOCAB = [
    "person",
    "man",
    "woman",
    "child",
    "table",
    "chair",
    "bed",
    "basket",
    "tool",
    "food",
    "bread",
    "fireplace",
    "window",
    "door",
    "cabinet",
]

NATURAL_OUTDOOR_OBJECT_VOCAB = [
    "person",
    "horse",
    "donkey",
    "cow",
    "sheep",
    "dog",
    "tree",
    "plant",
    "grass",
    "water",
    "rock",
    "path",
    "fence",
    "cart",
    "wagon"
]


GROUNDING_DINO_VOCABS = {
    "core": CORE_OBJECT_VOCAB,
    "rural_traditional": RURAL_OBJECT_VOCAB,
    "market_public": MARKET_OBJECT_VOCAB,
    "religious_heritage": RELIGIOUS_HERITAGE_OBJECT_VOCAB,
    "indoor_domestic": INDOOR_DOMESTIC_OBJECT_VOCAB,
    "natural_outdoor": NATURAL_OUTDOOR_OBJECT_VOCAB,
}


def build_prompt(vocab: list[str]) -> str:
    return ", ".join(vocab)


def select_grounding_vocab(scene_label: str | None = None) -> list[str]:
    if scene_label is None:
        return CORE_OBJECT_VOCAB

    groups = scene_groups_for_label(scene_label)

    if not groups:
        return CORE_OBJECT_VOCAB

    vocab = []
    for group in groups:
        vocab.extend(GROUNDING_DINO_VOCABS.get(group, []))

    if not vocab:
        return CORE_OBJECT_VOCAB

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(vocab))


def build_grounding_prompt_for_scene(scene_label: str | None = None) -> str:
    vocab = select_grounding_vocab(scene_label)
    return build_prompt(vocab)
