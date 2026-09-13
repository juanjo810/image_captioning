# Legacy

Este proyecto conserva piezas transicionales para reproducir experimentos anteriores y hacer ablations. No deben confundirse con la dirección principal: la refactorización hacia el core audioset-only.

## Qué es legacy hoy

| Área | Estado | Uso recomendado |
| --- | --- | --- |
| `CoreJSON` / `ExtendedJSON` (`--legacy-visual-core`) | Transicional | Comparación con experimentos previos y métricas estructuradas VG. |
| `acoustic_semantics.nodes` de Workflow A | Transicional | Solo con `--legacy-visual-core --include-audioset-nodes`. |
| Métricas visual-family handcrafted | Legacy/secundarias | Diagnóstico, no métrica principal. |
| `audioset_tag_*` y `audioset_category_*` | Alias retrocompatibles | Preferir `audioset_exact_node_*`, `audioset_parent_*` y `audioset_top_level_*`. |
| `--audioset-pred-source core_rules` | Baseline transicional | Reglas deterministas re-derivadas desde `core`. |
| `--audioset-pred-source vlm_nodes` | Transicional | Lee el bloque legacy `acoustic_semantics.nodes`. |
| `--audioset-pred-source union` | Ablation | Mezcla reglas y nodos VLM; reportar como ablation. |
| `evaluate_all_semantic_metrics.py` | Orquestador legacy | Solo con predicciones `CoreJSON`. |
| `scripts/run_json_captioning.py` (captioner Gemma) | Legacy | Se descartó el captioning con LLM: permitía "imaginar" evidencia. |

## Qué dejó de ser "planeado"

Estas cosas figuraban como TODO en versiones anteriores de la documentación y **ya están implementadas**:

- salida AudioSet canónica en `core.nodes` — es hoy el formato **por defecto** de ambos workflows
- vocabulario AudioSet-aware en Workflow B
- `--vocab-mode legacy|audioset|hybrid`

Y estas dejaron de existir:

- `Gemma4Adapter` / `--model gemma4` en Workflow A — eliminado; el único backend es llama.cpp
- `--temperature` en Workflow A — el sampling se configura en `llama-server`
- `AudioSetScene` (la escena como nodo AudioSet) — revertido; ver [audioset.md](audioset.md)
- `node_type` `scene_affordance` y `uncertain` en el core audioset — retirados; el `inference_type` del bloque legacy sí los mantiene

## Métricas legacy

Ejemplos:

- `entity_family_precision/recall/f1`
- `weighted_entity_family_precision/recall/f1`
- `discrete_family_precision/recall/f1`
- `weighted_discrete_family_precision/recall/f1`
- `interaction_family_precision/recall/f1`
- `family_count_bin_accuracy`

Agrupan etiquetas visuales en familias acústicamente relevantes. Son útiles para inspeccionar comportamiento, pero no son una métrica ontológica canónica.

En `filter_metrics_csv.py`, el grupo que antes se llamaba `soundscape` es ahora `legacy_soundscape`, precisamente para dejar constancia de que las métricas soundscape basadas en familias VG son secundarias.

## Cómo reportar experimentos nuevos

Prioriza las 5 métricas oficiales:

- `audioset_exact_node_f1`
- `audioset_parent_f1`
- `audioset_top_level_f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

Con `evaluate_audioset_semantics.py` + `filter_metrics_csv.py --metric-group audioset`, que usan `pred_source="core_nodes"` (lectura directa de `core.nodes`).

Si reportas una variante legacy, declara explícitamente la fuente:

```text
--audioset-pred-source core_rules
--audioset-pred-source vlm_nodes
--audioset-pred-source union
```

Y no describas Visual Genome como ground truth acústico: usa "AudioSet pseudo-references" o "acoustic-semantic pseudo-references".

## No borrar código legacy

No eliminar código legacy en tareas de documentación o evaluación comparativa sin una decisión explícita: puede ser necesario para reproducir resultados previos.
