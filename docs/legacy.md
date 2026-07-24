# Legacy

Este proyecto conserva piezas transicionales para comparar experimentos anteriores y hacer ablations. No deben confundirse con la dirección principal de la refactorización AudioSet.

## Qué es Legacy o Transicional

| Área | Estado | Uso recomendado |
| --- | --- | --- |
| Métricas visual-family handcrafted | Legacy/transicional | Diagnóstico, no métrica principal nueva. |
| `audioset_tag_*` y `audioset_category_*` | Alias backward-compatible | Preferir nombres `audioset_exact_node_*` y `audioset_top_level_*`. |
| Proyección visual-label-to-AudioSet por reglas | Implementada/transicional | Útil como baseline `core_rules`. |
| `core_rules` | Transicional/baseline | Compara reglas deterministas desde `core`. |
| `union` | Transicional/ablation | Mezcla reglas y nodos VLM; reportar como ablation. |
| Salidas visual-label canónicas antiguas | Legacy | No usarlas como evidencia AudioSet principal. |

## Métricas Legacy

Ejemplos:

- `entity_family_precision/recall/f1`
- `weighted_entity_family_precision/recall/f1`
- `discrete_family_precision/recall/f1`
- `weighted_discrete_family_precision/recall/f1`
- `interaction_family_precision/recall/f1`
- `family_count_bin_accuracy`

Estas métricas agrupan etiquetas visuales en familias acústicamente relevantes. Son útiles para inspeccionar comportamiento, pero no son una métrica ontológica canónica.

## AudioSet Actual vs Refactor Planeado

Implementado:

- ontología local en `ontology.json`
- loader en `metrics/audioset_ontology.py`
- métricas jerárquicas AudioSet
- `--audioset-pred-source core_rules|vlm_nodes|union`
- Workflow A opcional con `acoustic_semantics.nodes`

Planeado/TODO:

- salida AudioSet canónica en `core.nodes`
- vocabulario Workflow B AudioSet-aware
- `--vocab-mode legacy|audioset|hybrid`

## Cómo Reportar Experimentos Nuevos

Para experimentos nuevos, prioriza:

- `audioset_exact_node_f1`
- `audioset_parent_f1`
- `audioset_top_level_f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

Y declara explícitamente la fuente:

```text
--audioset-pred-source core_rules
--audioset-pred-source vlm_nodes
--audioset-pred-source union
```

No describas Visual Genome como ground truth acústico. Usa "AudioSet pseudo-references" o "acoustic-semantic pseudo-references".

## No Borrar Código Legacy

No eliminar código legacy en tareas de documentación o evaluación comparativa sin una decisión explícita. Puede ser necesario para reproducir resultados previos.
