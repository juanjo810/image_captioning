"""
vg_stratified_subset.py
=======================
Genera un subset estratificado de Visual Genome para evaluación acústica.

Estratificación en 3 ejes:
  1. Tipo de escena   : indoor | outdoor | nature
  2. Densidad social  : none (0 personas) | low (1-2) | high (3+)
  3. Complejidad      : simple (<10 objetos) | medium (10-25) | complex (>25)

Uso:
    python vg_stratified_subset.py \
        --objects   /ruta/a/objects.json \
        --image_data /ruta/a/image_data.json \
        --output    vg_subset_2000.json \
        --n         2000 \
        --seed      42

Salida:
    JSON con lista de image_ids seleccionados + sus metadatos de estrato.
"""

import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabularios heurísticos para clasificación acústica
# ---------------------------------------------------------------------------

INDOOR_OBJECTS = {
    "table", "chair", "couch", "sofa", "bed", "desk", "lamp", "shelf",
    "cabinet", "refrigerator", "oven", "sink", "toilet", "bathtub",
    "tv", "television", "monitor", "keyboard", "phone", "ceiling", "floor",
    "wall", "door", "window", "curtain", "pillow", "blanket", "carpet",
    "rug", "stove", "microwave", "dishwasher", "dryer", "washer",
    "bookshelf", "bookcase", "drawer", "mirror", "fan", "clock",
    "painting", "picture", "poster", "vase", "plate", "cup", "bowl",
    "glass", "bottle", "fork", "knife", "spoon",
}

NATURE_OBJECTS = {
    "tree", "grass", "flower", "mountain", "river", "lake", "ocean",
    "sea", "beach", "forest", "field", "meadow", "valley", "hill",
    "rock", "stone", "waterfall", "stream", "cloud", "sky", "sun",
    "moon", "star", "leaf", "branch", "bush", "shrub", "plant",
    "bird", "deer", "bear", "wolf", "fox", "rabbit", "squirrel",
    "butterfly", "insect", "fish", "whale", "dolphin", "sand", "soil",
    "mud", "snow", "ice", "horizon",
}

OUTDOOR_URBAN_OBJECTS = {
    "car", "truck", "bus", "bicycle", "motorcycle", "road", "street",
    "sidewalk", "building", "house", "skyscraper", "bridge", "fence",
    "traffic light", "stop sign", "fire hydrant", "bench", "pole",
    "sign", "parking", "crosswalk", "pavement", "train", "airplane",
    "boat", "ship", "van", "taxi", "ambulance",
}

PERSON_ALIASES = {
    "person", "people", "man", "woman", "boy", "girl", "child",
    "human", "kid", "baby", "adult", "teen", "teenager",
}


# ---------------------------------------------------------------------------
# Clasificadores
# ---------------------------------------------------------------------------

def classify_scene_type(obj_names: List[str]) -> str:
    """Clasifica la escena en 'indoor', 'outdoor' o 'nature' por voto mayoritario."""
    names_lower = {n.lower() for n in obj_names}

    votes = {
        "indoor":   sum(1 for n in names_lower if n in INDOOR_OBJECTS),
        "outdoor":  sum(1 for n in names_lower if n in OUTDOOR_URBAN_OBJECTS),
        "nature":   sum(1 for n in names_lower if n in NATURE_OBJECTS),
    }

    # En caso de empate, jerarquía: indoor > outdoor > nature
    max_votes = max(votes.values())
    if max_votes == 0:
        return "unknown"
    for label in ("indoor", "outdoor", "nature"):
        if votes[label] == max_votes:
            return label


def classify_social_density(obj_names: List[str]) -> str:
    """Cuenta personas presentes: none | low | high."""
    count = sum(1 for n in obj_names if n.lower() in PERSON_ALIASES)
    if count == 0:
        return "none"
    elif count <= 2:
        return "low"
    else:
        return "high"


def classify_complexity(n_objects: int) -> str:
    """Clasifica por número de objetos únicos anotados."""
    if n_objects < 10:
        return "simple"
    elif n_objects <= 25:
        return "medium"
    else:
        return "complex"


def get_stratum(scene: str, social: str, complexity: str) -> str:
    """Clave de estrato compuesta."""
    return f"{scene}|{social}|{complexity}"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_objects(path: Path) -> Dict[int, List[str]]:
    """
    Retorna {image_id: [nombre_objeto, ...]} con todos los nombres de objetos.
    Maneja tanto la versión raw (lista de imágenes) como alias aplanados.
    """
    log.info(f"Cargando {path} ...")
    with open(path, "r") as f:
        data = json.load(f)

    image_objects: Dict[int, List[str]] = {}

    # Formato estándar VG: lista de dicts con 'image_id' y 'objects'
    if isinstance(data, list) and "objects" in data[0]:
        for entry in data:
            iid = entry["image_id"]
            names = []
            for obj in entry["objects"]:
                names.extend(obj.get("names", []))
            image_objects[iid] = names

    # Formato aplanado (alias): lista de objetos con 'image_id' por objeto
    elif isinstance(data, list) and "image_id" in data[0]:
        for obj in data:
            iid = obj["image_id"]
            if iid not in image_objects:
                image_objects[iid] = []
            image_objects[iid].extend(obj.get("names", []))

    else:
        raise ValueError("Formato de objects.json no reconocido.")

    log.info(f"  → {len(image_objects)} imágenes cargadas.")
    return image_objects


def load_image_ids(path: Path) -> List[int]:
    """Carga image_data.json y devuelve lista de image_ids."""
    log.info(f"Cargando {path} ...")
    with open(path, "r") as f:
        data = json.load(f)
    ids = [entry["image_id"] for entry in data]
    log.info(f"  → {len(ids)} imágenes en image_data.")
    return ids


# ---------------------------------------------------------------------------
# Estratificación y muestreo
# ---------------------------------------------------------------------------

def build_strata(
    image_objects: Dict[int, List[str]]
) -> Dict[str, List[int]]:
    """Agrupa image_ids por clave de estrato."""
    strata: Dict[str, List[int]] = defaultdict(list)

    for iid, names in image_objects.items():
        scene      = classify_scene_type(names)
        social     = classify_social_density(names)
        complexity = classify_complexity(len(set(names)))
        key        = get_stratum(scene, social, complexity)
        strata[key].append(iid)

    return dict(strata)


def proportional_sample(
    strata: Dict[str, List[int]],
    n: int,
    seed: int,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Muestreo estratificado proporcional.
    Si un estrato tiene menos imágenes que las asignadas, usa todas.
    """
    rng = random.Random(seed)
    total = sum(len(v) for v in strata.values())
    selected: List[int] = []
    allocation: Dict[str, int] = {}

    # Primera pasada: cuota proporcional
    quotas = {}
    for key, ids in strata.items():
        quota = max(1, round(n * len(ids) / total))
        quotas[key] = quota

    # Ajuste para que sumen exactamente n
    diff = n - sum(quotas.values())
    # Añadir/quitar del estrato más grande
    largest_key = max(quotas, key=lambda k: len(strata[k]))
    quotas[largest_key] = max(1, quotas[largest_key] + diff)

    for key, ids in strata.items():
        k = min(quotas[key], len(ids))
        chosen = rng.sample(ids, k)
        selected.extend(chosen)
        allocation[key] = k

    return selected, allocation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera subset estratificado de Visual Genome para evaluación acústica."
    )
    parser.add_argument("--objects",    required=True,  help="Ruta a objects.json de VG")
    parser.add_argument("--image_data", required=True,  help="Ruta a image_data.json de VG")
    parser.add_argument("--output",     required=True,  help="Ruta del JSON de salida")
    parser.add_argument("--n",          type=int, default=2000, help="Tamaño del subset (default: 2000)")
    parser.add_argument("--seed",       type=int, default=42,   help="Semilla aleatoria (default: 42)")
    parser.add_argument(
        "--exclude_unknown",
        action="store_true",
        help="Excluir imágenes cuya escena no pudo clasificarse"
    )
    args = parser.parse_args()

    # 1. Cargar datos
    image_objects = load_objects(Path(args.objects))
    all_ids       = load_image_ids(Path(args.image_data))

    # Intersección: solo imágenes que aparecen en ambos ficheros
    valid_ids = set(all_ids) & set(image_objects.keys())
    log.info(f"Imágenes válidas (en ambos ficheros): {len(valid_ids)}")
    image_objects = {iid: image_objects[iid] for iid in valid_ids}

    # 2. Clasificar y agrupar en estratos
    log.info("Clasificando imágenes en estratos...")
    strata = build_strata(image_objects)

    if args.exclude_unknown:
        removed = strata.pop("unknown|none|simple", [])   # y otras combinaciones con unknown
        for key in [k for k in list(strata.keys()) if k.startswith("unknown")]:
            removed += strata.pop(key)
        log.info(f"Imágenes excluidas (unknown): {len(removed)}")

    # 3. Reporte de estratos
    log.info("\n── Distribución de estratos (completo) ──")
    for key in sorted(strata.keys()):
        scene, social, complexity = key.split("|")
        log.info(f"  {scene:8s} | social={social:4s} | {complexity:7s} → {len(strata[key]):6d} imgs")

    # 4. Muestreo estratificado
    log.info(f"\nMuestreando {args.n} imágenes (seed={args.seed})...")
    selected_ids, allocation = proportional_sample(strata, args.n, args.seed)

    # 5. Construir output con metadata de estrato por imagen
    id_to_stratum: Dict[int, str] = {}
    for key, ids in strata.items():
        for iid in ids:
            id_to_stratum[iid] = key

    output_records = []
    for iid in selected_ids:
        key = id_to_stratum[iid]
        scene, social, complexity = key.split("|")
        output_records.append({
            "image_id":   iid,
            "scene_type": scene,
            "social":     social,
            "complexity": complexity,
        })

    # 6. Guardar
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(
            {
                "subset_size":  len(output_records),
                "seed":         args.seed,
                "n_requested":  args.n,
                "strata_allocation": allocation,
                "images":       output_records,
            },
            f,
            indent=2,
        )

    # 7. Reporte final
    log.info(f"\n── Subset generado: {len(output_records)} imágenes ──")
    log.info(f"Guardado en: {out_path}")

    scene_counts = Counter(r["scene_type"] for r in output_records)
    social_counts = Counter(r["social"] for r in output_records)
    complexity_counts = Counter(r["complexity"] for r in output_records)

    log.info("\nDistribución por eje:")
    log.info(f"  Tipo de escena : {dict(scene_counts)}")
    log.info(f"  Densidad social: {dict(social_counts)}")
    log.info(f"  Complejidad    : {dict(complexity_counts)}")

    log.info("\n✓ Listo. Usa 'images' del JSON resultante para tu pipeline de evaluación.")


if __name__ == "__main__":
    main()