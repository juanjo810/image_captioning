# Repository Structure

Mapa práctico del repositorio.

```text
image_captioning/
  README.md
  ontology.json
  resources/
  requirements_workflow_a.txt
  requirements_workflowb.txt
  requirements_evaluation.txt
  src/
  metrics/
  scripts/
  docs/
```

## `src/`

Módulos compartidos por los dos workflows:

- `schemas.py` — los dos esquemas de salida (`AudioSetCoreJSON`/`AudioSetExtendedJSON` y los legacy `CoreJSON`/`ExtendedJSON`/`AcousticSemantics`). Todos los modelos heredan de `StrictBaseModel` (`extra="forbid"`).
- `geometry.py` — geometría por detección y agregados globales (ruta legacy de Workflow B).
- `fusion.py` — `build_from_modules`, punto de ensamblaje del `CoreJSON` legacy.
- `postprocessing.py` — `filter_detections`: confianza, NMS y dedup semántico entre batches.
- `captioning.py` — `build_audioset_caption` (ruta por defecto) y `build_caption` (ruta legacy). Ambos deterministas, sin LLM.

## `src/workflow_a/`

Pipeline VLM:

- `prompt_builder.py` — un builder por modo de llamada y por ruta (`single`, `three`, `two`, `five`, legacy)
- `parser.py` — extracción del bloque JSON de la respuesta cruda
- `validator.py` — normalización y validación Pydantic de cada formato
- `pipeline.py` — orquesta llamadas, escribe outputs y el manifest JSONL
- `audioset_nodes.py` — allow-list de nodos AudioSet y de términos visuales
- `base.py` + `adapters/` — interfaz `BaseVLM` y `LlamaCppServerAdapter` (único backend)

Entrypoints:

```text
scripts/run_workflow_a.py
scripts/run_workflow_a_with_retries.py
```

## `src/workflow_b/`

Pipeline modular visual:

- `grounding_dino_adapter.py`, `owlv2_adapter.py` — detectores open-vocabulary
- `places365_adapter.py`, `places365_mapping.py` — clasificación de escena y allow-list de etiquetas
- `vocabularies.py`, `audioset_vocab.py` — batches de prompts (universales, por escena, AudioSet)
- `audioset_projection.py` — proyección de detecciones a `AudioSetCoreJSON` (ruta por defecto)
- `hoi_adapter.py`, `upt_adapter.py`, `hoi_fusion.py`, `hoi_utils.py`, `dummy_adapters.py` — HOI (ruta legacy)
- `spatial_relations.py`, `semantic_salience.py` — relaciones espaciales e importancia semántica (ruta legacy)
- `constants.py` — `SCENE_GROUPS` y `CATEGORY_MAP`, las taxonomías de las que dependen fusión y salience

Entrypoint:

```text
scripts/run_grounding_dino_pipeline.py
```

## `metrics/`

- `audioset_ontology.py` — carga e indexa `ontology.json` (DAG).
- `audioset_leaf_vocab.py` — parsea `resources/audioset_leaf_node_names.txt`: nodos permitidos, términos detectables y reglas hoja.
- `audioset_semantics.py` — proyección a nodos AudioSet y las métricas jerárquicas.
- `embedding_metrics.py` — similitud por embeddings de frases (sentence-transformers).

## `scripts/`

Entrypoints de ejecución. Todos se invocan como módulo (`python -m scripts.<nombre>`):

- `run_workflow_a.py`, `run_workflow_a_with_retries.py`
- `run_grounding_dino_pipeline.py`
- `run_dummy_pipeline.py` — smoke test del contrato de fusión, sin modelos
- `run_json_captioning.py` — captioner Gemma sobre JSON ya generados (legacy)
- `run_grounding_dino.py`, `run_places365.py` — scripts de un solo módulo, para depuración
- `run_pilot_workflow_b.sh`, `run_pilot_gemma.sh` — barridos piloto

## `scripts/evaluation/`

- `build_prediction_manifest.py`, `build_vg_references.py` — manifests y referencias
- `validate_prediction_jsons.py`, `validate_audioset_ontology.py` — validación previa
- `evaluate_audioset_semantics.py` — las 5 métricas oficiales
- `evaluate_scene_awareness.py` — coherencia de escena, válida para ambos formatos
- `evaluate_all_caption_metrics.py` y sus componentes (`evaluate_caption_metrics.py`, `evaluate_clipscore.py`, `evaluate_chair.py`)
- `evaluate_all_semantic_metrics.py`, `evaluate_structured_vg.py`, `evaluate_soundscape_semantics.py` — solo formato legacy
- `filter_metrics_csv.py` — recorte a las columnas oficiales de un grupo
- `vg_utils.py` — fontanería compartida: alias, canonicalización, carga de predicciones, `is_audioset_core_json`
- `compare_detection_counts.py` — diagnóstico de detecciones

## `resources/`

`audioset_leaf_node_names.txt`: la lista hoja de la que salen el vocabulario de detección, la allow-list del VLM y las reglas de evaluación.

## `docs/`

Esta documentación. Solo ficheros `.md`.

## `outputs/`, `data/`, checkpoints

Directorios locales esperados, no versionados. No versionar:

- checkpoints de GroundingDINO, Places365 y UPT
- cachés de Hugging Face
- datasets de Visual Genome
- outputs JSON/CSV generados

## `legacy/`

No existe tal directorio. La lógica legacy vive dentro de los módulos y scripts actuales, detrás del flag `--legacy-visual-core` y de los scripts marcados como tales. Ver [legacy.md](legacy.md).
