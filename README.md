# image_captioning

Repositorio de investigación para convertir imágenes en representaciones semánticas estructuradas y evaluar esas salidas contra referencias visuales y pseudo-referencias acústico-semánticas basadas en la AudioSet ontology.

El proyecto contiene dos rutas principales:

- **Workflow A**: usa un VLM para generar directamente un JSON `core` y, opcionalmente, nodos AudioSet inferidos visualmente.
- **Workflow B**: usa Places365, un detector open-vocabulary (GroundingDINO u OWLv2), postprocesado geométrico y HOI opcional para construir `core`, `extended` y una caption.

Importante: este repositorio no reconoce audio real desde imágenes. AudioSet se usa como vocabulario semántico para razonar sobre sonidos plausibles a partir de evidencia visual.

## Estado Actual

| Área | Estado | Nota |
| --- | --- | --- |
| Workflow A `core` JSON | Implementado | CLI en `scripts/run_workflow_a.py`. |
| Workflow A nodos AudioSet | Parcial | Se escriben en `acoustic_semantics.nodes` con `--include-audioset-nodes`; todavía no son el `core.nodes` canónico planeado. |
| Workflow B con GroundingDINO | Implementado | Requiere repo/checkpoint externos y rutas locales esperadas por el script. |
| Workflow B con OWLv2 | Implementado | Descarga/carga modelo Hugging Face; ignora `text_threshold`. |
| HOI en Workflow B | Parcial | `dummy` para smoke tests y `upt` como backend local; depende de UPT/checkpoints externos. |
| Vocabulario Workflow B AudioSet-aware | Planeado/TODO | No existe `--vocab-mode` todavía. |
| Evaluación semántica AudioSet | Implementada | Usa pseudo-referencias desde Visual Genome, no ground truth acústico. |
| Métricas legacy visual-family | Legacy/transicional | Útiles como diagnóstico, no como evidencia principal nueva. |

## Arquitectura Conceptual

```text
Imagen
  |-- Workflow A: VLM -> JSON core -> caption -> acoustic_semantics opcional
  |
  `-- Workflow B: Places365 -> vocabulario visual -> GroundingDINO/OWLv2
        -> filtros + geometría -> HOI opcional -> JSON core/extended -> caption

Evaluación:
predicciones JSON + manifest CSV + referencias Visual Genome
  -> métricas VG estructuradas
  -> métricas semantic/soundscape legacy
  -> métricas AudioSet ontology con pseudo-referencias
```

## Estructura

```text
src/workflow_a/       Pipeline VLM y validación de JSON.
src/workflow_b/       Detectores, vocabularios, Places365, HOI y fusión.
metrics/              Métricas y utilidades AudioSet.
scripts/              Entrypoints de ejecución.
scripts/evaluation/   Construcción de manifests/referencias y evaluación.
docs/                 Documentación extendida.
ontology.json         AudioSet ontology local.
```

Los directorios `data/`, `outputs/` y checkpoints/modelos externos son esperados localmente, pero no deberían versionarse.

## Instalación

La instalación oficial usa tres entornos Conda separados:

| Entorno | Uso | Versiones verificadas |
| --- | --- | --- |
| `imagecap-a` | Workflow A / VLM / llama.cpp client | Python 3.11, Torch 2.11.0+cu130 |
| `imagecap-b` | Workflow B / GroundingDINO / OWLv2 / Places365 / HOI | Python 3.10, Torch 2.5.1+cu118 |
| `imagecap-eval` | Métricas, Visual Genome, AudioSet, SPICE | Python 3.10, Torch 2.12.0+cu130 |

No se recomienda fusionarlos: los modelos y librerías externas tienen requisitos incompatibles de Python, PyTorch y CUDA.

Sigue el manual completo en [docs/setup.md](docs/setup.md).

## Modelos Externos

- **Workflow A llama.cpp**: servidor OpenAI-compatible en `http://localhost:8889` por defecto.
- **Workflow A Gemma**: modelo Hugging Face `google/gemma-4-E4B-it` por defecto.
- **Workflow B GroundingDINO**: repo local y checkpoint `.pth` bajo el layout recomendado de `/home/jovyan/projects`.
- **Workflow B OWLv2**: `google/owlv2-base-patch16-ensemble`.
- **Workflow B Places365**: `categories_places365.txt` y checkpoint por arquitectura.
- **Workflow B UPT**: repo/checkpoint/dataset externos si se usa `--hoi upt`.

## Smoke Tests

```bash
python -m py_compile scripts/run_workflow_a.py scripts/run_grounding_dino_pipeline.py scripts/evaluation/evaluate_all_semantic_metrics.py
python scripts/evaluation/validate_audioset_ontology.py
```

## Ejecutar Workflow A

Una imagen con llama.cpp:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Directorio de imágenes `.jpg` y nodos AudioSet opcionales:

```bash
python scripts/run_workflow_a.py \
  --image-dir data/images \
  --limit 20 \
  --model llamacpp \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

Detalles: [docs/workflow_a.md](docs/workflow_a.md).

## Ejecutar Workflow B

Con OWLv2:

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector owlv2 \
  --hoi none \
  --output-dir outputs/workflow_b/owlv2_resnet50
```

Con GroundingDINO:

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --hoi none \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

El script actual espera modelos bajo `/home/jovyan/projects`. Ver [docs/workflow_b.md](docs/workflow_b.md).

## Evaluación Semántica

```bash
python scripts/evaluation/evaluate_all_semantic_metrics.py \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --audioset-pred-source core_rules \
  --output outputs/metrics/semantic_summary.csv \
  --per-image-output outputs/metrics/semantic_per_image.csv
```

Ver [docs/evaluation.md](docs/evaluation.md) y [docs/data_and_manifests.md](docs/data_and_manifests.md).

## AudioSet

AudioSet se usa como ontología canónica para pseudo-referencias acústico-semánticas. Las métricas principales actuales son:

- `audioset_exact_node_precision/recall/f1`
- `audioset_parent_precision/recall/f1`
- `audioset_top_level_precision/recall/f1`
- `audioset_lca_similarity`
- `audioset_tree_distance_similarity`

Más contexto: [docs/audioset.md](docs/audioset.md).

## Outputs Esperados

- Workflow A: `raw/`, `json/`, `captions/`, `manifests/manifest.jsonl`, `failed/`.
- Workflow B: JSON por imagen si se pasa `--output-dir`; si no, imprime en stdout.
- Evaluación: CSV summary y, opcionalmente, CSV per-image.

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
- Visual Genome no contiene etiquetas acústicas; se usan pseudo-referencias.
- Workflow B tiene rutas externas hardcoded bajo `/home/jovyan/projects`.
- `--vocab-mode legacy|audioset|hybrid` está planeado, no implementado.
- La salida AudioSet canónica en `core.nodes` está planeada; hoy Workflow A usa `acoustic_semantics.nodes`.
- La instalación requiere entornos separados; no mezclar Workflow A, Workflow B y Evaluation.

## Nota de Investigación

Este repositorio está en estado interno/de investigación. Úsalo para experimentación reproducible y análisis, no como paquete estable de producción.
