# image_captioning

Repositorio de investigación para convertir imágenes en representaciones semánticas estructuradas y evaluar esas salidas contra referencias visuales (Visual Genome) y pseudo-referencias acústico-semánticas basadas en la AudioSet ontology.

El proyecto contiene dos rutas independientes que **no interoperan** entre sí:

- **Workflow A**: un VLM (servido por llama.cpp) genera directamente el JSON estructurado a partir de la imagen. Sin detectores.
- **Workflow B**: Places365 + detector open-vocabulary (GroundingDINO u OWLv2) + postprocesado determinista. Sin VLM.

Importante: este repositorio **no reconoce audio real** desde imágenes. AudioSet se usa como vocabulario y ontología semántica para razonar sobre sonidos *plausibles* a partir de evidencia visual.

## Dos esquemas de salida

Ambos workflows emiten por defecto el esquema **audioset-only**, y ambos aceptan `--legacy-visual-core` para volver al esquema **legacy**:

| Esquema | Flag | `core` contiene |
| --- | --- | --- |
| `AudioSetCoreJSON` (por defecto) | — | `image_id`, `scene`, `nodes[]`, `caption` |
| `CoreJSON` (legacy) | `--legacy-visual-core` | `image_id`, `scene`, `entities[]`, `observed_interactions[]`, `spatial_relations[]`, `environment`, `caption` |

`scene` es el **mismo modelo `Scene`** en los dos esquemas (`label` de Places365, `indoor_outdoor`, `confidence` opcional).

Cuando se pasa `--legacy-visual-core`, la salida se escribe en un subdirectorio `legacy/` de `--output-dir` en ambos workflows.

Definiciones exactas: [src/schemas.py](src/schemas.py). Todos los modelos usan `extra="forbid"`, así que cualquier campo inesperado falla la validación en vez de colarse.

## Estado Actual

| Área | Estado | Nota |
| --- | --- | --- |
| Workflow A `AudioSetCoreJSON` | Implementado | Por defecto. Modos `--call-mode single\|three\|two`. |
| Workflow A `CoreJSON` legacy | Implementado | Solo con `--legacy-visual-core`. |
| Workflow A backend VLM | Solo llama.cpp | `LlamaCppServerAdapter`. El adaptador Gemma vía Transformers fue eliminado. |
| Workflow B `AudioSetCoreJSON` | Implementado | Proyección por reglas hoja, sin HOI ni relaciones espaciales. |
| Workflow B `CoreJSON` legacy | Implementado | Solo con `--legacy-visual-core` (geometría + HOI + fusión). |
| Workflow B con GroundingDINO | Implementado | Requiere repo/checkpoint externos. |
| Workflow B con OWLv2 | Implementado | Descarga modelo Hugging Face; ignora `text_threshold`. |
| HOI en Workflow B | Implementado | `dummy` para smoke tests, `upt` como backend real. Solo en la ruta legacy. |
| `--vocab-mode legacy\|audioset\|hybrid` | Implementado | Solo se lee del CLI en la ruta legacy; la ruta por defecto fuerza `audioset`. |
| Evaluación AudioSet | Implementada | 5 métricas oficiales, sobre pseudo-referencias VG. |
| Métricas legacy visual-family | Legacy/secundarias | Diagnóstico, no evidencia principal. |

## Arquitectura Conceptual

```text
Imagen
  |-- Workflow A: VLM -> visual_terms -> nodes AudioSet + scene + caption
  |                     (1, 2 o 3 llamadas según --call-mode)
  |
  `-- Workflow B: Places365 -> batches de vocabulario AudioSet
        -> GroundingDINO/OWLv2 -> filtros/NMS/dedup
        -> proyección por reglas hoja -> nodes + scene + caption determinista

Evaluación:
predicciones JSON + manifest CSV + referencias Visual Genome
  -> métricas de caption (CIDEr/SPICE, CLIPScore, CHAIR)
  -> 5 métricas AudioSet ontológicas (pseudo-referencias)
  -> scene awareness (independiente, vale para ambos esquemas)
  -> métricas estructuradas VG y soundscape legacy (solo esquema legacy)
```

## Estructura

```text
src/workflow_a/       Pipeline VLM: prompts, parser, validador, adaptadores.
src/workflow_b/       Detectores, Places365, vocabularios, HOI, proyección AudioSet.
src/                  geometry.py, fusion.py, postprocessing.py, captioning.py, schemas.py
metrics/              Ontología AudioSet, semántica AudioSet, embeddings.
scripts/              Entrypoints de ejecución.
scripts/evaluation/   Manifests, referencias VG y scripts de métricas.
resources/            audioset_leaf_node_names.txt (vocabulario hoja compartido).
docs/                 Documentación extendida.
ontology.json         AudioSet ontology local (DAG).
```

Los directorios `data/`, `outputs/` y los checkpoints/modelos externos son locales y no se versionan.

## Instalación

La instalación oficial usa tres entornos Conda separados:

| Entorno | Uso | Versiones verificadas |
| --- | --- | --- |
| `imagecap-a` | Workflow A / cliente llama.cpp | Python 3.11 |
| `imagecap-b` | Workflow B / GroundingDINO / OWLv2 / Places365 / HOI | Python 3.10, Torch 2.5.1+cu118 |
| `imagecap-eval` | Métricas, Visual Genome, AudioSet, SPICE | Python 3.10, Torch 2.12.0+cu130 |

No deben fusionarse: los modelos y librerías externas tienen requisitos incompatibles de Python, PyTorch y CUDA.

Manual completo: [docs/setup.md](docs/setup.md).

## Cómo se ejecutan los scripts

**Siempre como módulo, desde la raíz del repositorio**:

```bash
python -m scripts.run_workflow_a ...
```

`python scripts/run_workflow_a.py` **falla** con `ModuleNotFoundError: No module named 'src.workflow_a'`, porque Python pone `scripts/` en el path en vez de la raíz del repo.

## Modelos Externos

- **Workflow A**: servidor llama.cpp OpenAI-compatible, por defecto en `http://localhost:8889`. Modelos GGUF servidos por `llama-server` (`--server-url` apunta a él). El sampling (temperatura, top-p) se configura **en el servidor**: el adaptador solo envía `model`, `messages` y `max_tokens`.
- **Workflow B GroundingDINO**: repo local importable y checkpoint `.pth` bajo `/home/jovyan/projects`.
- **Workflow B OWLv2**: `google/owlv2-base-patch16-ensemble`.
- **Workflow B Places365**: `categories_places365.txt` y checkpoint por arquitectura.
- **Workflow B UPT**: repo/checkpoint/dataset externos, solo si se usa `--hoi upt` con `--legacy-visual-core`.

## Smoke Tests

```bash
python -m py_compile scripts/run_workflow_a.py scripts/run_grounding_dino_pipeline.py scripts/evaluation/evaluate_all_semantic_metrics.py
python -m scripts.evaluation.validate_audioset_ontology
python -m scripts.run_dummy_pipeline   # no necesita modelos: ejercita el contrato de fusión con fixtures
```

## Ejecutar Workflow A

Una imagen (salida audioset por defecto):

```bash
python -m scripts.run_workflow_a \
  --image data/images/example.jpg \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Directorio de imágenes `.jpg`, con la ruta legacy y nodos acústicos:

```bash
python -m scripts.run_workflow_a \
  --image-dir data/images \
  --limit 20 \
  --legacy-visual-core \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

Modos de llamada alternativos (solo en la ruta audioset):

```bash
python -m scripts.run_workflow_a --image data/images/example.jpg --call-mode three
python -m scripts.run_workflow_a --image data/images/example.jpg --call-mode two
```

Lotes desatendidos con reintento automático de las imágenes que fallaron:

```bash
python -m scripts.run_workflow_a_with_retries \
  --image-dir data/images \
  --server-url http://localhost:8889 \
  --max-retries 2
```

Detalles: [docs/workflow_a.md](docs/workflow_a.md).

## Ejecutar Workflow B

Salida audioset por defecto (GroundingDINO):

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

Lote sobre un directorio (los modelos se cargan una sola vez):

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image-dir data/images \
  --limit 100 \
  --detector owlv2 \
  --scene-architecture densenet161 \
  --output-dir outputs/workflow_b/owlv2_densenet161
```

Ruta legacy con HOI:

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --legacy-visual-core \
  --hoi upt \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

El script espera modelos bajo `/home/jovyan/projects`. Ver [docs/workflow_b.md](docs/workflow_b.md).

## Evaluación

Las 5 métricas AudioSet oficiales, en dos pasos. Primero el scorer, que escribe el CSV completo (las 5 métricas más precision/recall y columnas diagnósticas):

```bash
python -m scripts.evaluation.evaluate_audioset_semantics \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --output outputs/metrics/audioset_summary.csv
```

Después el recorte a las columnas oficiales, que es lo que se reporta:

```bash
python -m scripts.evaluation.filter_metrics_csv \
  --input outputs/metrics/audioset_summary.csv \
  --output outputs/metrics/audioset_official.csv \
  --metric-group audioset
```

`--metric-group` acepta `caption`, `chair`, `scene`, `audioset` y `legacy_soundscape`. Para la escena, el recorte se hace sobre el CSV de `evaluate_scene_awareness.py` con `--metric-group scene`.

Métricas de caption (CIDEr/SPICE, CLIPScore, CHAIR) en un solo CSV:

```bash
python -m scripts.evaluation.evaluate_all_caption_metrics \
  --manifest outputs/manifests/predictions.csv \
  --caption-refs data/vg_refs/vg_caption_references.csv \
  --image-dir data/images \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --output outputs/metrics/caption_summary.csv
```

`evaluate_all_semantic_metrics.py` es **solo para el esquema legacy** (lee `core.entities` y `core.observed_interactions`). Ver [docs/evaluation.md](docs/evaluation.md) y [docs/data_and_manifests.md](docs/data_and_manifests.md).

## AudioSet

AudioSet se usa como ontología canónica para pseudo-referencias acústico-semánticas. Las 5 métricas oficiales son:

- `audioset_exact_node_f1`
- `audioset_parent_f1`
- `audioset_top_level_f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

(Precision y recall de las tres primeras también se escriben.) La coherencia escena/objetos se evalúa aparte y **no** cuenta como una de las 5. Más contexto: [docs/audioset.md](docs/audioset.md).

## Outputs Esperados

- **Workflow A**: `raw/`, `json/`, `captions/`, `manifests/manifest.jsonl`, `failed/` bajo `--output-dir` (o bajo `<output-dir>/legacy/` con `--legacy-visual-core`).
- **Workflow B**: `<output-dir>/<image_id>.json` (o `<output-dir>/legacy/<image_id>.json`). Sin `--output-dir` imprime a stdout.
- **Evaluación**: CSV summary por `(detector, scene_model, captioner)` y, opcionalmente, CSV per-image.

## Mapa de Documentación

- [Setup](docs/setup.md)
- [Workflow A](docs/workflow_a.md)
- [Workflow B](docs/workflow_b.md)
- [AudioSet](docs/audioset.md)
- [Evaluation](docs/evaluation.md)
- [Data and manifests](docs/data_and_manifests.md)
- [Repository structure](docs/repository_structure.md)
- [Legacy](docs/legacy.md)
- [Troubleshooting](docs/troubleshooting.md)

## Limitaciones Conocidas

- No hay reconocimiento acústico real.
- Visual Genome no contiene etiquetas acústicas; se usan pseudo-referencias derivadas de objetos y relaciones.
- Workflow B tiene rutas externas hardcoded bajo `/home/jovyan/projects`.
- En Workflow A, un nodo sin `visual_evidence_terms` válidos tras la validación se descarta; la caption se escribió antes de validar, así que puede mencionar un sonido que ya no tiene nodo asociado.
- Workflow A no emite `extended`: sus predicciones no tienen geometría ni grounding.
- `evaluate_all_semantic_metrics.py` apuntado a predicciones `AudioSetCoreJSON` devuelve ceros en vez de avisar: sus métricas leen `core.entities`/`core.observed_interactions` con un default de lista vacía, y el esquema audioset no tiene esos campos, así que precision/recall/F1 salen 0.0 sin ningún error. Es un resultado inválido que parece válido: úsalo solo con predicciones `--legacy-visual-core`.
- La instalación requiere entornos separados; no mezclar Workflow A, Workflow B y Evaluation.

## Nota de Investigación

Este repositorio está en estado interno/de investigación. Úsalo para experimentación reproducible y análisis, no como paquete estable de producción.
