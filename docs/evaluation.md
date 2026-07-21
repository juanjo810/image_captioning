# Evaluation

La evaluación semántica unificada combina métricas Visual Genome estructuradas, métricas soundscape/scene awareness y métricas AudioSet basadas en pseudo-referencias.

Script principal:

```bash
python scripts/evaluation/evaluate_all_semantic_metrics.py --help
```

## CLI Real

Argumentos implementados:

| Argumento | Requerido | Descripción |
| --- | --- | --- |
| `--manifest` | sí | CSV de predicciones. |
| `--vg-object-refs` | sí | CSV con objetos VG por imagen. |
| `--vg-relationship-refs` | sí | CSV con relaciones VG por imagen. |
| `--object-alias` | sí | CSV/mapa de alias de objetos. |
| `--relationship-alias` | sí | CSV/mapa de alias de relaciones. |
| `--output` | sí | CSV resumen agregado. |
| `--label-space` | no | `workflow_b` u `open`; default `workflow_b`. |
| `--per-image-output` | no | CSV opcional por imagen. |
| `--audioset-pred-source` | no | `core_rules`, `vlm_nodes` o `union`; default `core_rules`. |

## Comando de Ejemplo

```bash
python scripts/evaluation/evaluate_all_semantic_metrics.py \
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

## Inputs Requeridos

### Manifest CSV

El loader usa `csv.DictReader` y espera, como mínimo:

```text
image_id,json_path,detector,scene_model,captioner
```

`json_path` debe apuntar al JSON de predicción. `detector`, `scene_model` y `captioner` se usan para agrupar el summary.

### Visual Genome Object References

CSV con:

```text
image_id,objects
```

`objects` contiene etiquetas separadas por `|`.

### Visual Genome Relationship References

CSV con:

```text
image_id,relationships
```

`relationships` contiene tripletas separadas por `|`, cada una con:

```text
subject::predicate::object
```

### Alias Maps

Los alias se cargan con `load_alias_map` desde `scripts/evaluation/vg_utils.py`. Deben normalizar variantes de objetos y relaciones a una forma canónica usada por las métricas.

## AudioSet Prediction Source

`--audioset-pred-source` controla de dónde salen las predicciones AudioSet:

| Valor | Estado | Qué usa |
| --- | --- | --- |
| `core_rules` | Implementado | Reglas deterministas desde `core.scene`, `core.entities` y `core.observed_interactions`. |
| `vlm_nodes` | Implementado | Lee `acoustic_semantics.nodes` generado por Workflow A. |
| `union` | Implementado/transicional | Unión de reglas `core` y nodos VLM. Útil para ablation. |

Las tres variantes se restringen al conjunto hoja compartido definido en `resources/audioset_leaf_node_names.txt` y parseado por `metrics/audioset_leaf_vocab.py`.

## Métricas AudioSet Principales

Estas son las métricas recomendadas para reportar en experimentos nuevos:

- `audioset_exact_node_precision`
- `audioset_exact_node_recall`
- `audioset_exact_node_f1`
- `audioset_parent_precision`
- `audioset_parent_recall`
- `audioset_parent_f1`
- `audioset_top_level_precision`
- `audioset_top_level_recall`
- `audioset_top_level_f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

También se escriben aliases legacy como `audioset_tag_f1` y `audioset_category_f1` por compatibilidad.

## Summary vs Per-Image

- `--output`: CSV agregado por `(detector, scene_model, captioner)`.
- `--per-image-output`: CSV con métricas e información diagnóstica por imagen.

El summary omite campos diagnósticos largos como listas de tags y paths de nodos.

## Métricas Legacy

El evaluador conserva métricas como:

- `entity_family_*`
- `discrete_family_*`
- `weighted_discrete_family_*`
- `interaction_family_*`
- `scene_family_known`
- `background_family_known`

Son útiles para diagnóstico y comparación histórica. Para resultados nuevos centrados en AudioSet, trátalas como secundarias.

## Nota Sobre Visual Genome

Visual Genome no contiene ground truth acústico. Las referencias AudioSet se generan como pseudo-referencias a partir de objetos y relaciones visuales. Usa los términos **AudioSet pseudo-references** o **acoustic-semantic pseudo-references**.
