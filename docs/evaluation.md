# Evaluation

La evaluación puntúa predicciones (JSON + caption de cualquiera de los dos workflows) contra referencias, en dos ejes en gran medida independientes: **calidad de la caption** y **calidad semántica/estructurada**.

Todos los scripts se ejecutan como módulo desde la raíz del repo y, normalmente, en el entorno `imagecap-eval`:

```bash
python -m scripts.evaluation.<script> --help
```

## Qué script sirve para qué formato

Esta es la decisión más importante antes de evaluar nada: cuál es el esquema de tus predicciones.

| Script | `AudioSetCoreJSON` | `CoreJSON` legacy |
| --- | --- | --- |
| `evaluate_audioset_semantics.py` (5 métricas oficiales) | Sí | Sí |
| `evaluate_scene_awareness.py` | Sí | Sí |
| `evaluate_all_caption_metrics.py` | Sí (ver CHAIR abajo) | Sí |
| `evaluate_all_semantic_metrics.py` | **No** | Sí |
| `evaluate_structured_vg.py`, `evaluate_soundscape_semantics.py` | **No** | Sí |
| `build_prediction_manifest.py`, `validate_prediction_jsons.py` | Sí (rama propia) | Sí |

`evaluate_all_semantic_metrics.py` y los scripts que orquesta leen `core.entities` y `core.observed_interactions` con un default de lista vacía. Apuntados a predicciones `AudioSetCoreJSON` **no fallan**: devuelven precision/recall/F1 = 0.0, que parece un resultado válido y no lo es. Úsalos solo con predicciones generadas con `--legacy-visual-core`.

## Las 5 métricas oficiales AudioSet

Son las métricas ontológicas que se reportan en los experimentos nuevos:

- `audioset_exact_node_f1`
- `audioset_parent_f1`
- `audioset_top_level_f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

(El scorer también escribe precision y recall de las tres primeras, más columnas diagnósticas.)

El scorer dedicado es `evaluate_audioset_semantics.py`, deliberadamente separado de `evaluate_all_semantic_metrics.py`: solo llama a `compute_audioset_metrics` + `aggregate` + `write_csv`.

```bash
python -m scripts.evaluation.evaluate_audioset_semantics \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --output outputs/metrics/audioset_summary.csv \
  --per-image-output outputs/metrics/audioset_per_image.csv
```

Ese CSV lleva todas las columnas. El recorte a las oficiales se hace en un segundo paso con `filter_metrics_csv.py` (ver abajo).

La coherencia escena/objetos **no** es una de las 5: se puntúa aparte con `evaluate_scene_awareness.py`.

### De dónde salen predicciones y referencias

`metrics/audioset_semantics.py::predicted_audioset_tag_counts` toma un `pred_source`, hoy con default `"core_nodes"`:

| Valor | Qué lee |
| --- | --- |
| `core_nodes` (default) | `core.nodes[].audioset_id` directamente del `AudioSetCoreJSON`, vía `core_nodes_tag_counts`. |
| `core_rules` (legacy) | Reglas deterministas re-derivadas en tiempo de evaluación desde `core.entities`/`core.observed_interactions`. |
| `vlm_nodes` (legacy) | El bloque legacy `acoustic_semantics.nodes` de Workflow A. |
| `union` (legacy) | Unión de `core_rules` y `vlm_nodes`. Ablation. |

`core_nodes_tag_counts` **no toca `core.scene`** en ningún caso.

`reference_audioset_tag_counts` deriva las referencias únicamente desde objetos y relaciones de Visual Genome, con las mismas reglas hoja.

`evaluate_audioset_semantics.py` no expone flag para cambiar el `pred_source`; usa el default. El flag `--audioset-pred-source` (`core_rules`, `vlm_nodes`, `union`) solo existe en el orquestador legacy `evaluate_all_semantic_metrics.py`.

## Scene awareness

`evaluate_scene_awareness.py` es el único script semántico que funciona sin cambios contra **ambos** formatos: ramifica con `is_audioset_core_json`, leyendo `core.nodes[].visual_evidence_terms` en la ruta audioset y `core.entities[].count_estimate` en la legacy. Por eso se mantiene fuera de `evaluate_all_semantic_metrics.py` y se ejecuta por separado.

```bash
python -m scripts.evaluation.evaluate_scene_awareness \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --output outputs/metrics/scene_summary.csv \
  --per-image-output outputs/metrics/scene_per_image.csv
```

Columnas principales: `scene_group_known`, `pred_entity_scene_consistency`, `scene_confidence`.

`scene_confidence` reporta `"n/a"` por imagen cuando `core.scene.confidence` no existe (siempre, en Workflow A). El agregado trata `""` y `"n/a"` como ausentes, así que solo cae a `""` cuando ningún elemento del grupo tenía confianza real, en vez de promediar un cero falso.

## Métricas de caption

- `evaluate_caption_metrics.py` — CIDEr y SPICE vía `pycocoevalcap` (SPICE necesita Java).
- `evaluate_clipscore.py` — alineación imagen-caption con CLIP.
- `evaluate_chair.py` — tasa de alucinación, contra los objetos de referencia de VG y contra el propio JSON estructurado.
- `evaluate_all_caption_metrics.py` — orquesta las tres en un único CSV por configuración `(detector, scene_model, captioner)`.

```bash
python -m scripts.evaluation.evaluate_all_caption_metrics \
  --manifest outputs/manifests/predictions.csv \
  --caption-refs data/vg_refs/vg_caption_references.csv \
  --image-dir data/images \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --output outputs/metrics/caption_summary.csv
```

`--clip-model-name` por defecto es `openai/clip-vit-base-patch16`.

### CHAIR y el formato audioset

`CHAIRi_VG` se calcula para toda fila, sea cual sea el esquema: las referencias VG son ground truth externo, independiente del formato de la predicción.

`CHAIRi_JSON` y `avg_hallucinated_json` solo tienen sentido contra `core.entities`, que un `AudioSetCoreJSON` no tiene. Para esas filas `load_json_entities` devuelve `None` (no un conjunto vacío) y la fila se **excluye** del promedio, en vez de contarla como 100 % de alucinación. Si ninguna fila del grupo tenía entidades, la celda del CSV sale vacía, no `0.0`.

## Orquestador legacy

`evaluate_all_semantic_metrics.py` combina métricas VG estructuradas, soundscape/scene-awareness legacy y AudioSet. **Solo para el esquema legacy.**

| Argumento | Requerido | Descripción |
| --- | --- | --- |
| `--manifest` | sí | CSV de predicciones. |
| `--vg-object-refs` | sí | CSV con objetos VG por imagen. |
| `--vg-relationship-refs` | sí | CSV con relaciones VG por imagen. |
| `--object-alias` | sí | Mapa de alias de objetos. |
| `--relationship-alias` | sí | Mapa de alias de relaciones. |
| `--output` | sí | CSV resumen agregado. |
| `--label-space` | no | `workflow_b` u `open`; default `workflow_b`. |
| `--per-image-output` | no | CSV opcional por imagen. |
| `--audioset-pred-source` | no | `core_rules`, `vlm_nodes` o `union`; default `core_rules`. |

```bash
python -m scripts.evaluation.evaluate_all_semantic_metrics \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --label-space workflow_b \
  --audioset-pred-source core_rules \
  --output outputs/metrics/semantic_summary.csv \
  --per-image-output outputs/metrics/semantic_per_image.csv
```

Conserva métricas legacy como `entity_family_*`, `discrete_family_*`, `weighted_discrete_family_*`, `interaction_family_*`, `scene_family_known` y `background_family_known`. Son diagnóstico y comparación histórica, no evidencia principal. Ver [legacy.md](legacy.md).

## Recorte a las columnas oficiales

`filter_metrics_csv.py` recorta un CSV de métricas a las columnas oficiales de un grupo. Falla si falta alguna columna, en vez de escribir un CSV incompleto.

```bash
python -m scripts.evaluation.filter_metrics_csv \
  --input outputs/metrics/audioset_summary.csv \
  --output outputs/metrics/audioset_official.csv \
  --metric-group audioset
```

| `--metric-group` | Columnas (además de `detector`, `scene_model`, `captioner`, `n`) |
| --- | --- |
| `caption` | `CIDEr`, `SPICE` |
| `chair` | `CHAIRi_VG`, `CHAIRi_JSON`, `avg_caption_objects_mentioned`, `avg_hallucinated_vg`, `avg_hallucinated_json` |
| `scene` | `scene_group_known`, `pred_entity_scene_consistency` |
| `audioset` | las 5 oficiales |
| `legacy_soundscape` | `weighted_entity_family_f1`, `weighted_discrete_family_f1`, `acoustic_weighted_discrete_presence_recall`, `interaction_family_f1` |

El grupo `audioset` exporta las 5 métricas ontológicas (antes exportaba los alias retrocompatibles `audioset_tag_*` / `audioset_category_*`). El antiguo grupo `soundscape` se llama ahora `legacy_soundscape`, documentando que las métricas soundscape basadas en familias VG son secundarias.

El grupo `scene` se aplica al CSV de `evaluate_scene_awareness.py`, no al de AudioSet: son ficheros distintos.

## Validación previa

Antes de puntuar, conviene validar los JSON de predicción contra el esquema:

```bash
python -m scripts.evaluation.validate_prediction_jsons \
  --input outputs/pilot_vg/visual_json_audioset/grounding_dino_resnet50 \
  --output outputs/metrics/validation.csv
```

También ramifica con `is_audioset_core_json`. Para predicciones audioset comprueba:

- consistencia de `extended.grounding[].node_id` contra `core.nodes`
- `audioset_id_valid_ratio` — que cada `audioset_id` resuelva en la ontología
- `visual_evidence_terms_valid_ratio` — que los `visual_evidence_terms` de cada nodo sean subconjunto del vocabulario detectable compartido, contado aparte del ratio anterior

`visual_evidence_terms_valid_ratio` debería ser siempre 1.0 en predicciones generadas después de la restricción de `node_type` en Workflow A: el validador de generación descarta el nodo entero en vez de emitirlo con evidencia vacía. Un valor por debajo de 1.0 solo aparece con predicciones anteriores a ese cambio.

Para predicciones legacy comprueba en cambio la consistencia `entity_id` / `extended_ids`.

## Inputs requeridos

### Manifest CSV

El loader usa `csv.DictReader` y necesita como mínimo:

```text
image_id,json_path,detector,scene_model,captioner
```

`json_path` apunta al JSON de predicción; `detector`, `scene_model` y `captioner` agrupan el summary. Construye el manifest con `build_prediction_manifest.py` — ver [data_and_manifests.md](data_and_manifests.md).

### Referencias Visual Genome

```text
vg_object_references.csv        image_id,objects            # objetos separados por |
vg_relationship_references.csv  image_id,relationships      # tripletas subject::predicate::object separadas por |
vg_caption_references.csv       image_id,reference_caption  # una fila por caption de referencia
```

Se generan con `build_vg_references.py`. Ojo con un detalle de los dumps de VG: el script lee solo `names` en subject/object de las relaciones, pero VG guarda `name` (singular) en la gran mayoría. Si no normalizas `name` → `names` al preparar los datos, `vg_relationship_references.csv` sale casi vacío.

### Alias maps

Se cargan con `load_alias_map` de `scripts/evaluation/vg_utils.py`, en formato `canonical,alias1,alias2,...`. Los ficheros `object_alias.txt` y `relationship_alias.txt` que distribuye Visual Genome ya vienen en ese formato.

`vg_utils.py` es la fontanería compartida (canonicalización de etiquetas, mapas de alias, carga de predicciones, extracción de entidades/interacciones, `is_audioset_core_json`). Revísalo antes de añadir un script de métricas nuevo, para no reimplementar la normalización.

## Summary vs per-image

- `--output`: CSV agregado por `(detector, scene_model, captioner)`.
- `--per-image-output`: CSV por imagen, con información diagnóstica.

El summary omite campos diagnósticos largos como listas de tags y rutas de nodos.

## Nota sobre Visual Genome

Visual Genome no contiene ground truth acústico. Las referencias AudioSet se generan como **pseudo-referencias** proyectando objetos y relaciones visuales sobre la ontología. Usa los términos "AudioSet pseudo-references" o "acoustic-semantic pseudo-references", nunca "ground truth acústico".
