# Workflow B

Workflow B construye la representación estructurada desde módulos visuales deterministas: Places365 para la escena, GroundingDINO u OWLv2 para detección open-vocabulary, y postprocesado. **No hay VLM en el bucle.**

Los pasos 1–3 son comunes; a partir del paso 4 la ruta depende de `--legacy-visual-core`:

- **por defecto (audioset core)**: `AudioSetCoreJSON` + `AudioSetExtendedJSON`, por proyección de las detecciones sobre reglas hoja AudioSet. Sin HOI, sin relaciones espaciales, sin `build_from_modules`.
- **`--legacy-visual-core`**: `CoreJSON` + `ExtendedJSON` clásicos, con geometría, HOI opcional, relaciones espaciales y fusión. La salida va a un subdirectorio `legacy/` de `--output-dir`.

## Estado

| Componente | Estado |
| --- | --- |
| Places365 scene classification | Implementado |
| GroundingDINO detector | Implementado, requiere instalación externa |
| OWLv2 detector | Implementado |
| Proyección AudioSet (`AudioSetCoreJSON`) | Implementado, ruta por defecto |
| Filtros, geometría y fusión `core`/`extended` legacy | Implementado (`--legacy-visual-core`) |
| HOI `dummy` | Implementado, solo smoke tests de fusión |
| HOI `upt` | Implementado, depende de repo/checkpoints externos |
| `--vocab-mode legacy\|audioset\|hybrid` | Implementado |
| Modo lote `--image-dir` | Implementado (los modelos se cargan una sola vez) |

## CLI Real

```bash
python -m scripts.run_grounding_dino_pipeline --help
```

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--image` | `/home/jovyan/projects/data/test.jpg` | Imagen de entrada. |
| `--image-dir` | `None` | Directorio con imágenes: procesa cada `*.jpg` en vez de `--image`. |
| `--limit` | `None` | Solo con `--image-dir`: tope de imágenes procesadas. |
| `--detector` | `grounding_dino` | `grounding_dino` u `owlv2`. |
| `--vocab-mode` | `legacy` | `legacy`, `audioset` o `hybrid`. **Solo se lee en la ruta legacy** (ver abajo). |
| `--scene-architecture` | `resnet50` | `resnet50` o `densenet161`. |
| `--box-threshold` | `None` | Si se omite, se usa el threshold propio de cada batch de vocabulario. |
| `--text-threshold` | `None` | Ídem. Relevante para GroundingDINO; OWLv2 lo ignora. |
| `--min-confidence` | `0.30` | Filtro posterior de confianza en `filter_detections`. |
| `--nms-iou-threshold` | `0.70` | IoU de NMS en `filter_detections`. |
| `--hoi` | `none` | `none`, `dummy` o `upt`. Solo tiene efecto con `--legacy-visual-core`. |
| `--output-dir` | `None` | Si se omite, imprime el JSON en stdout. |
| `--output-name` | `None` | Nombre del JSON de salida. Por defecto, stem de la imagen + `.json`. |
| `--verbose` | off | Log de detecciones crudas, filtradas y HOI. |
| `--legacy-visual-core` | off | Usa el `CoreJSON` clásico en lugar del audioset core. |

Notas importantes sobre los defaults:

- `--box-threshold` y `--text-threshold` valen `None` a propósito. Sin valor CLI, cada batch de prompts aporta el suyo; si pasas un valor por CLI, ese gana para todos los batches. Los valores efectivamente usados quedan registrados en `metadata`.
- `--min-confidence 0.30` está por encima del threshold efectivo del detector, así que sí filtra de verdad.
- `--vocab-mode` solo se respeta con `--legacy-visual-core`. En la ruta por defecto el script fuerza `vocab_mode="audioset"` independientemente del flag.
- `--hoi` cambió su default de `upt` a `none`, y solo significa algo en la ruta legacy.

## Flujo Actual

```text
image
  -> Places365Adapter.predict(topk=5)
  -> iter_grounding_prompt_batches(scene["label"], vocab_mode efectivo)
  -> GroundingDINOAdapter u OWLv2Adapter, un pase por batch
  -> filter_detections(min_confidence, NMS, dedup semántico por IoU/containment)
  |
  |-- por defecto: project_detections_to_audioset
  |     -> nodes (reglas hoja) + scene Places365 + build_audioset_caption
  |     -> AudioSetCoreJSON + AudioSetExtendedJSON
  |
  `-- --legacy-visual-core: build_from_modules
        -> geometría + HOI opcional + relaciones espaciales + build_caption
        -> CoreJSON + ExtendedJSON
```

## Places365

Places365 predice la escena y se usa para:

- `core.scene.label`
- `core.scene.confidence` — el score real del clasificador.
- inferir `core.scene.indoor_outdoor`
- **solo en modo `legacy` o `hybrid`**: elegir batches de vocabulario adicionales según el grupo de escena

En la ruta por defecto (audioset), la etiqueta de escena **no** influye en qué se detecta: el vocabulario es el mismo para todas las imágenes. Places365 ahí solo rellena `core.scene`.

Rutas esperadas:

```text
/home/jovyan/projects/places365/categories_places365.txt
/home/jovyan/projects/models/places365/<architecture>_places365.pth.tar
```

## Vocabulario y batches de prompts

El detector no recibe un único prompt gigante: `iter_grounding_prompt_batches(scene_label, vocab_mode)` genera una **secuencia de batches**, y el script hace un pase de detección por batch, acumulando todas las detecciones crudas antes de filtrar. Cada batch trae su propio prompt y sus propios thresholds.

Qué batches se generan depende del `vocab_mode` efectivo:

| `vocab_mode` | Batches generados | ¿Depende de la escena? |
| --- | --- | --- |
| `audioset` (forzado en la ruta por defecto) | 7 batches de hasta 30 términos cada uno, cubriendo los 210 términos detectables de las reglas hoja AudioSet (`box_threshold` 0.35, `text_threshold` 0.25) | No. Idénticos para toda imagen. |
| `legacy` | 6 batches universales (`UNIVERSAL_GROUNDING_PROMPT_BATCHES`) + un batch por grupo de escena que tenga vocabulario en `SCENE_EXPANSION_VOCABS` (`box_threshold` 0.28, `text_threshold` 0.22) | Sí, en la parte de expansión. |
| `hybrid` | los batches `legacy` seguidos de los `audioset` | Sí, en la parte legacy. |

Los grupos de escena (`scene_groups_for_label`, definidos en `SCENE_EXPANSION_VOCABS`) son 13: `coastal_water`, `education_health_office`, `entertainment_culture`, `garden_park`, `indoor_domestic`, `industrial_workshop`, `market_public`, `natural_outdoor`, `public_indoor`, `religious_heritage`, `rural_traditional`, `sports_recreation`, `urban_transport`. Una etiqueta de Places365 puede caer en varios grupos, o en ninguno: si no hay vocabulario para el grupo, no se emite batch extra.

Los términos son **visuales** en todos los modos: personas, animales, vehículos, estructuras, naturaleza, objetos, herramientas, comida. Los detectores ven evidencia visual, no audio.

Los términos AudioSet se derivan de `resources/audioset_leaf_node_names.txt` mediante `metrics/audioset_leaf_vocab.py`, la misma fuente que usa Workflow A para su vocabulario de `visual_terms` y de la que salen las reglas de evaluación. Se excluyen los nodos marcados como `no`, los del bloque `FROM HERE` / `TO HERE`, los nodos `blacklist` de la ontología y las líneas sin evidencia estructurada. Que generación y evaluación compartan este vocabulario hoja es lo que mantiene consistente el espacio acústico entre ambas.

## Detectores

### GroundingDINO

Repo local importable como `groundingdino` y checkpoint:

```text
/home/jovyan/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
```

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

### OWLv2

Usa `google/owlv2-base-patch16-ensemble` con `transformers`.

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector owlv2 \
  --output-dir outputs/workflow_b/owlv2_resnet50
```

OWLv2 convierte el prompt del batch en candidate labels. `text_threshold` se conserva por compatibilidad de firma, pero se ignora: OWLv2 no expone el mismo concepto de umbral textual por token.

### Modo lote

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image-dir data/subsets/vg_test_100/images \
  --limit 100 \
  --detector grounding_dino \
  --scene-architecture densenet161 \
  --output-dir outputs/pilot_vg/visual_json_audioset/grounding_dino_densenet161
```

El bucle ocurre dentro del proceso: los modelos se cargan una sola vez. Un proceso por configuración, no uno por imagen. `scripts/run_pilot_workflow_b.sh [audioset|legacy]` automatiza el barrido `{grounding_dino, owlv2} x {resnet50, densenet161}`.

## Postprocesado

`postprocessing.filter_detections` recibe todas las detecciones crudas de todos los batches y aplica, en este orden: filtro por `--min-confidence`, NMS con `--nms-iou-threshold`, y dedup semántico entre batches por IoU (`0.30`) y containment (`0.65`). Ese dedup es necesario precisamente porque el mismo objeto suele aparecer en varios batches.

`metadata` registra los cuatro valores resueltos más `n_raw_detections` y `n_filtered_detections`, que es la forma rápida de ver si los thresholds están cortando de más o de menos.

## Ruta por defecto: proyección AudioSet

`src/workflow_b/audioset_projection.py::project_detections_to_audioset`:

- empareja las detecciones filtradas contra `metrics/audioset_leaf_vocab.py::audioset_leaf_rules`, las mismas reglas hoja que se usan para derivar las referencias de evaluación;
- construye cada `AudioSetCoreNode`, rellenando `visual_evidence_terms` con las etiquetas normalizadas de las detecciones que dispararon la regla, más `confidence`, `parent_ids` (`AudioSetOntology.parents_or_self`) y `top_level_ids` (`top_levels`). Workflow A no rellena ninguno de estos tres últimos;
- rellena `core.visual_terms` con las etiquetas normalizadas de **todas** las detecciones filtradas, no solo las que dispararon una regla hoja — un objeto detectado sin regla acústica asociada sigue siendo un objeto VG válido, y antes quedaba invisible en la caption;
- construye la escena como un `Scene` de Places365 normal (`label` / `indoor_outdoor` vía `infer_indoor_outdoor_from_scene` / `confidence` real). **La escena no es un nodo AudioSet**;
- emite `AudioSetExtendedJSON` con `grounding[]` (bbox, `node_id`, `detector_confidence`, `source`) y `global_geometry`.

`caption` y `acoustic_caption` se escriben por separado, ambos deterministas y sin LLM:

- `caption` viene de `src/captioning.py::build_audioset_caption(scene, visual_terms)`: solo la frase visual, a partir de `core.visual_terms` (hasta 8), sin tocar los nodos.
- `acoustic_caption` viene de `build_audioset_acoustic_caption(scene, nodes, node_instance_counts)`: un sonido por nodo, rankeado por confianza, con singular/plural vía `_audioset_node_phrase`. `None` si `nodes` está vacío.

`visual_evidence_terms` de cada nodo sigue saliendo solo de las detecciones que dispararon su regla — sigue siendo, por construcción, subconjunto de `visual_terms`.

No hay HOI, ni `spatial_relations.py`, ni `src/fusion.py` en esta ruta.

## Ruta legacy: `--legacy-visual-core`

- `geometry.py` — geometría por detección (ratio de área de bbox, tamaño relativo, posición, centralidad, salience) y agregados globales
- HOI opcional (`upt_adapter.py` con el modelo UPT, o `DummyHOIAdapter` para smoke tests) → `hoi_fusion.py` empareja cajas HOI con entidades por IoU para producir `observed_interactions`
- `spatial_relations.py` deriva relaciones left/right/above/below entre entidades
- `src/fusion.py::build_from_modules` agrupa las detecciones por instancia en `Entity` semánticas (agregando duplicados, calculando `count_estimate`), calcula `semantic_importance` (`semantic_salience.py`, a partir del grupo de escena y la pertenencia a interacciones) y emite `CoreJSON` + `ExtendedJSON`
- `src/captioning.py::build_caption` — caption determinista por plantilla, ensamblada desde entidades rankeadas, interacciones verbalizadas y relaciones espaciales. Es template-based a propósito, para que la caption siga siendo trazable a campos concretos del JSON durante la evaluación

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --legacy-visual-core \
  --hoi upt \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

`src/workflow_b/constants.py` contiene `SCENE_GROUPS` y `CATEGORY_MAP`, las taxonomías de las que dependen la fusión y la salience

## HOI

- `--hoi none` (default): sin Human-Object Interaction.
- `--hoi dummy`: tripleta hardcoded para validar el contrato de fusión.
- `--hoi upt`: `UPTAdapter` con rutas externas (`upt/`, `models/upt/upt-r50-hicodet.pt`, `upt/hicodet`).

Solo surte efecto junto con `--legacy-visual-core`.

## Output JSON

Ruta por defecto:

```json
{
  "core": {
    "image_id": "example",
    "scene": { "label": "street", "indoor_outdoor": "outdoor", "confidence": 0.61 },
    "visual_terms": ["car", "person", "traffic light"],
    "nodes": [
      {
        "node_id": "n1",
        "audioset_id": "/m/012f08",
        "audioset_name": "Motor vehicle (road)",
        "node_type": "visible_source",
        "evidence": "...",
        "visual_evidence_terms": ["car"],
        "confidence": 0.44,
        "parent_ids": ["/m/07yv9"],
        "top_level_ids": ["/t/dd00041"]
      }
    ],
    "caption": "The image shows a street scene with car, person, and traffic light.",
    "acoustic_caption": "Motor vehicle noise would plausibly be heard."
  },
  "extended": {
    "grounding": [],
    "global_geometry": {}
  },
  "metadata": {
    "workflow": "B",
    "detector": "owlv2",
    "vocab_mode": "legacy",
    "scene_model": {},
    "box_threshold": 0.35,
    "text_threshold": 0.25,
    "min_confidence": 0.3,
    "nms_iou_threshold": 0.7,
    "n_raw_detections": 41,
    "n_filtered_detections": 12,
    "ontology_mode": "dag"
  }
}
```

En la ruta legacy, `core` es el `CoreJSON` clásico (`entities`, `observed_interactions`, `spatial_relations`, `environment`), `extended` tiene `entities_extended` + `global_geometry`, y `metadata` sustituye `ontology_mode` por `hoi_backend` y `raw_hoi_count`.

Ojo: `metadata.vocab_mode` registra el valor **del CLI**, no el efectivo. En la ruta por defecto el vocabulario usado siempre es `audioset` aunque ahí ponga `legacy`.

Sin `--output-dir`, el JSON se imprime en stdout. Con él, se guarda como `<image_stem>.json` (o `--output-name`) dentro de `--output-dir`, o dentro de `<output-dir>/legacy/` si se pasó `--legacy-visual-core`.
