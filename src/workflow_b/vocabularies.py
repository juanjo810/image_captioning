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

COASTAL_WATER_SCENES = {
    "beach",
    "coast",
    "ocean",
    "harbor",
    "pier",
    "canal natural",
    "canal urban",
    "lagoon",
    "lake natural",
    "river",
    "creek",
    "pond",
    "waterfall",
    "watering hole",
    "boathouse",
    "boat deck",
}

PUBLIC_INDOOR_SCENES = {
    "auditorium",
    "atrium public",
    "banquet hall",
    "conference center",
    "conference room",
    "lobby",
    "waiting room",
    "library indoor",
    "museum indoor",
    "stage indoor",
    "restaurant",
    "cafeteria",
    "bar",
    "pub indoor",
}

EDUCATION_HEALTH_OFFICE_SCENES = {
    "classroom",
    "kindergarden classroom",
    "lecture room",
    "office",
    "office cubicles",
    "computer room",
    "hospital",
    "hospital room",
    "operating room",
    "biology laboratory",
    "chemistry lab",
    "physics laboratory",
}

SPORTS_RECREATION_SCENES = {
    "soccer field",
    "football field",
    "baseball field",
    "stadium soccer",
    "stadium football",
    "stadium baseball",
    "athletic field outdoor",
    "playground",
    "basketball court indoor",
    "gymnasium indoor",
    "ski slope",
    "ski resort",
    "swimming pool outdoor",
    "swimming pool indoor",
}

INDUSTRIAL_WORKSHOP_SCENES = {
    "assembly line",
    "auto factory",
    "construction site",
    "industrial area",
    "junkyard",
    "landfill",
    "engine room",
    "repair shop",
    "loading dock",
    "hardware store",
}

GARDEN_PARK_SCENES = {
    "botanical garden",
    "formal garden",
    "japanese garden",
    "zen garden",
    "topiary garden",
    "roof garden",
    "greenhouse outdoor",
    "greenhouse indoor",
    "patio",
    "picnic area",
    "gazebo exterior",
}

ENTERTAINMENT_CULTURE_SCENES = {
    "amphitheater",
    "arena performance",
    "ballroom",
    "discotheque",
    "movie theater indoor",
    "music studio",
    "orchestra pit",
    "stage outdoor",
    "art gallery",
    "art studio",
    "museum outdoor",
}

SCENE_GROUPS = {
    "rural_traditional": RURAL_TRADITIONAL_SCENES,
    "natural_outdoor": NATURAL_OUTDOOR_SCENES,
    "market_public": MARKET_PUBLIC_SCENES,
    "religious_heritage": RELIGIOUS_HERITAGE_SCENES,
    "indoor_domestic": INDOOR_DOMESTIC_SCENES,
    "urban_transport": URBAN_TRANSPORT_SCENES,

    "coastal_water": COASTAL_WATER_SCENES,
    "public_indoor": PUBLIC_INDOOR_SCENES,
    "education_health_office": EDUCATION_HEALTH_OFFICE_SCENES,
    "sports_recreation": SPORTS_RECREATION_SCENES,
    "industrial_workshop": INDUSTRIAL_WORKSHOP_SCENES,
    "garden_park": GARDEN_PARK_SCENES,
    "entertainment_culture": ENTERTAINMENT_CULTURE_SCENES,
}

OUTDOOR_SCENE_GROUPS = {
    "rural_traditional",
    "natural_outdoor",
    "market_public",
    "religious_heritage",
    "urban_transport",
    "coastal_water",
    "sports_recreation",
    "garden_park",
    "entertainment_culture",
    "industrial_workshop",
}

INDOOR_SCENE_GROUPS = {
    "indoor_domestic",
    "public_indoor",
    "education_health_office",
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
    normalized = normalize_scene_label(scene_label)

    if "indoor" in normalized:
        return "indoor"

    if "outdoor" in normalized:
        return "outdoor"

    groups = set(scene_groups_for_label(scene_label))

    if groups & INDOOR_SCENE_GROUPS:
        return "indoor"

    if groups & OUTDOOR_SCENE_GROUPS:
        return "outdoor"

    return "unknown"


# ---------------------------------------------------------------------
# Grounding DINO vocabularies
# ---------------------------------------------------------------------

UNIVERSAL_GROUNDING_PROMPT_BATCHES = {
    "humans": {
        "terms": [
            "person",
            "man",
            "woman",
            "child",
            "crowd",
        ],
        "box_threshold": 0.20,
        "text_threshold": 0.20,
    },

    "animals": {
        "terms": [
            "horse",
            "donkey",
            "cow",
            "sheep",
            "goat",
            "dog",
            "cat",
            "bird",
        ],
        "box_threshold": 0.25,
        "text_threshold": 0.20,
    },

    "vehicles_transport": {
        "terms": [
            "car",
            "bus",
            "truck",
            "bicycle",
            "motorcycle",
            "cart",
            "wagon",
            "boat",
            "train",
            "tram",
            "wheelchair",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },

    "structures_built_environment": {
        "terms": [
            "building",
            "house",
            "church",
            "tower",
            "wall",
            "fence",
            "gate",
            "door",
            "window",
            "bridge",
            "street lamp",
            "bench",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },

    "nature_terrain": {
        "terms": [
            "tree",
            "plant",
            "grass",
            "flower",
            "crop",
            "field",
            "road",
            "street",
            "path",
            "water",
            "river",
            "rock",
            "sky",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },

    "objects_tools_food": {
        "terms": [
            "table",
            "chair",
            "basket",
            "bag",
            "bucket",
            "barrel",
            "box",
            "tool",
            "instrument",
            "umbrella",
            "food",
            "bread",
            "fruit",
            "vegetables",
        ],
        "box_threshold": 0.30,
        "text_threshold": 0.25,
    },
}

SCENE_EXPANSION_VOCABS = {
    # ----------------------------------
    # RURAL / TRADITIONAL
    # ----------------------------------
    "rural_traditional": [
        "barn",
        "stable",
        "corral",
        "fence",
        "hay",
        "haystack",
        "plow",
        "farm tool",
        "wooden cart",
        "wagon",
        "tractor",
        "field crop",
        "grain",
        "animal pen",
    ],

    # ----------------------------------
    # NATURAL OUTDOOR
    # ----------------------------------
    "natural_outdoor": [
        "tree",
        "forest",
        "bush",
        "grass",
        "rock",
        "river",
        "water",
        "waterfall",
        "lake",
        "mountain",
        "trail",
        "path",
        "log",
    ],

    # ----------------------------------
    # MARKET / PUBLIC
    # ----------------------------------
    "market_public": [
        "market stall",
        "vendor",
        "stand",
        "crate",
        "basket",
        "fruit",
        "vegetable",
        "awning",
        "table",
        "crowd",
        "sign",
    ],

    # ----------------------------------
    # RELIGIOUS / HERITAGE
    # ----------------------------------
    "religious_heritage": [
        "cathedral",
        "church",
        "temple",
        "altar",
        "statue",
        "arch",
        "column",
        "bell tower",
        "stone wall",
        "grave",
        "monument",
    ],

    # ----------------------------------
    # URBAN TRANSPORT
    # ----------------------------------
    "urban_transport": [
        "car",
        "bus",
        "truck",
        "tram",
        "bicycle",
        "motorcycle",
        "traffic light",
        "crosswalk",
        "sidewalk",
        "road",
        "street sign",
        "bridge",
        "rail",
    ],

    # ----------------------------------
    # INDOOR DOMESTIC
    # ----------------------------------
    "indoor_domestic": [
        "table",
        "chair",
        "bed",
        "sofa",
        "cabinet",
        "lamp",
        "window",
        "door",
        "kitchen appliance",
        "cooking pot",
        "bowl",
        "sink",
    ],

    # ----------------------------------
    # PUBLIC INDOOR
    # ----------------------------------
    "public_indoor": [
        "chair",
        "table",
        "stage",
        "screen",
        "counter",
        "desk",
        "sign",
        "queue barrier",
        "display",
    ],

    # ----------------------------------
    # EDUCATION / OFFICE / HEALTH
    # ----------------------------------
    "education_health_office": [
        "desk",
        "computer",
        "monitor",
        "chair",
        "whiteboard",
        "book",
        "bed",
        "medical equipment",
        "cabinet",
    ],

    # ----------------------------------
    # SPORTS / RECREATION
    # ----------------------------------
    "sports_recreation": [
        "ball",
        "goal",
        "net",
        "court",
        "field",
        "bench",
        "helmet",
        "equipment",
    ],

    # ----------------------------------
    # INDUSTRIAL / WORKSHOP
    # ----------------------------------
    "industrial_workshop": [
        "machine",
        "tool",
        "engine",
        "pipe",
        "metal structure",
        "crane",
        "forklift",
        "container",
    ],

    # ----------------------------------
    # COASTAL / WATER
    # ----------------------------------
    "coastal_water": [
        "boat",
        "ship",
        "dock",
        "pier",
        "water",
        "wave",
        "sand",
        "lifeguard tower",
    ],

    # ----------------------------------
    # GARDEN / PARK
    # ----------------------------------
    "garden_park": [
        "tree",
        "flower",
        "bench",
        "path",
        "grass",
        "fountain",
        "gazebo",
    ],

    # ----------------------------------
    # ENTERTAINMENT / CULTURE
    # ----------------------------------
    "entertainment_culture": [
        "stage",
        "instrument",
        "speaker",
        "light",
        "screen",
        "audience",
        "microphone",
    ],
}

def build_prompt_from_terms(terms: list[str]) -> str:
    return ", ".join(terms)


def iter_universal_prompt_batches():
    """Yield reproducible open-vocabulary detection batches.

    Batching avoids very long prompts and improves recall for small or frequent
    entities such as people in street scenes.
    """
    for batch_name, batch in UNIVERSAL_GROUNDING_PROMPT_BATCHES.items():
        yield {
            "name": batch_name,
            "prompt": build_prompt_from_terms(batch["terms"]),
            "box_threshold": batch["box_threshold"],
            "text_threshold": batch["text_threshold"],
        }

def iter_scene_expansion_prompt_batches(scene_label: str | None):
    """Yield scene-aware prompt batches derived from Places365 scene groups."""
    if scene_label is None:
        return

    groups = scene_groups_for_label(scene_label)

    for group in groups:
        terms = SCENE_EXPANSION_VOCABS.get(group)

        if not terms:
            continue

        yield {
            "name": f"scene_expansion:{group}",
            "prompt": build_prompt_from_terms(terms),
            "box_threshold": 0.28,
            "text_threshold": 0.22,
        }

def iter_grounding_prompt_batches(scene_label: str | None = None):
    """Yield universal + optional scene-aware Grounding DINO prompts."""
    yield from iter_universal_prompt_batches()

    if scene_label is not None:
        yield from iter_scene_expansion_prompt_batches(scene_label)
