# Repository Structure

Mapa práctico del repositorio.

```text
image_captioning/
  README.md
  ontology.json
  requirements_workflow_a.txt
  requirements_workflowb.txt
  requirements_evaluation.txt
  src/
  metrics/
  scripts/
  docs/
```

## `src/workflow_a/`

Pipeline VLM:

- construcción del prompt
- adaptadores VLM
- parsing de JSON
- validación Pydantic
- escritura de outputs
- lista permitida de nodos AudioSet opcionales

Entrypoint relacionado:

```text
scripts/run_workflow_a.py
```

## `src/workflow_b/`

Pipeline modular visual:

- adaptadores GroundingDINO y OWLv2
- Places365
- vocabularios visuales universales y scene-aware
- HOI adapters
- fusión de entidades/interacciones
- relaciones espaciales y salience

Entrypoint relacionado:

```text
scripts/run_grounding_dino_pipeline.py
```

## `metrics/`

Métricas y utilidades:

- `audioset_ontology.py`: carga e indexa `ontology.json`.
- `audioset_semantics.py`: mapeos visuales a pseudo-etiquetas AudioSet y métricas jerárquicas.
- `embedding_metrics.py`: métricas de embeddings/captions.

## `scripts/`

Entrypoints de ejecución:

- `run_workflow_a.py`
- `run_grounding_dino_pipeline.py`
- `run_json_captioning.py`
- scripts piloto y variantes históricas

## `scripts/evaluation/`

Herramientas de evaluación:

- construcción de manifests
- construcción de referencias VG
- evaluación semántica unificada
- métricas de caption
- validación de ontología

## `docs/`

Documentación del proyecto. El archivo `.docx` existente en `docs/` es un artefacto documental previo; los Markdown actuales son la documentación navegable para GitHub.

## `legacy/`

No hay directorio `legacy/` en el estado actual del repo. Hay, sin embargo, lógica legacy/transicional conservada en métricas y scripts. Ver [legacy.md](legacy.md).

## `outputs/`

Directorio esperado para resultados generados. No debe versionarse.

## `data/`

Directorio esperado para imágenes, Visual Genome, alias maps y referencias. No está comprometido en el repo.

## Checkpoints y Modelos

No versionar:

- checkpoints GroundingDINO
- checkpoints Places365
- modelos UPT
- cachés Hugging Face
- datasets Visual Genome
- outputs JSON/CSV generados a gran escala
