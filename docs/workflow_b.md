# Workflow B

Workflow B construye una representación estructurada desde módulos visuales: Places365 para escena, GroundingDINO u OWLv2 para detección open-vocabulary, filtros/geometry/fusion, HOI opcional y caption template.

## Estado

| Componente | Estado |
| --- | --- |
| Places365 scene classification | Implementado |
| GroundingDINO detector | Implementado, requiere instalación externa |
| OWLv2 detector | Implementado |
| Filtros y fusión `core`/`extended` | Implementado |
| HOI `dummy` | Implementado para smoke tests |
| HOI `upt` | Parcial, depende de repo/checkpoints externos |
| `--vocab-mode legacy|audioset|hybrid` | Implementado |
| Salida AudioSet canónica | Planeada/TODO |

## CLI Real

Script:

```bash
python scripts/run_grounding_dino_pipeline.py --help
```

Argumentos implementados:

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--image` | `/home/jovyan/projects/data/test.jpg` | Imagen de entrada. |
| `--detector` | `grounding_dino` | `grounding_dino` u `owlv2`. |
| `--vocab-mode` | `legacy` | `legacy`, `audioset` o `hybrid`. |
| `--scene-architecture` | `resnet50` | `resnet50` o `densenet161`. |
| `--box-threshold` | `0.30` | Umbral de caja pasado al detector. |
| `--text-threshold` | `0.25` | Umbral textual para GroundingDINO. |
| `--min-confidence` | `0.20` | Filtro posterior de confianza mínima. |
| `--hoi` | `upt` | `none`, `dummy` o `upt`. |
| `--output-dir` | `None` | Si se omite, imprime JSON en stdout. |
| `--output-name` | `None` | Nombre del JSON de salida. |
| `--verbose` | off | Logs de detecciones y HOI. |

Nota: aunque `--box-threshold` y `--text-threshold` existen en el CLI, los batches de vocabulario actuales incluyen sus propios thresholds. En el flujo actual, el detector recibe los valores del batch generado por `iter_grounding_prompt_batches`, mientras `metadata` guarda los valores CLI.

## Flujo Actual

```text
image
  -> Places365Adapter.predict(topk=5)
  -> iter_grounding_prompt_batches(scene["label"], vocab_mode)
  -> GroundingDINOAdapter u OWLv2Adapter
  -> filter_detections(min_confidence, NMS, semantic filters)
  -> build_from_modules(core, extended)
  -> infer_indoor_outdoor_from_scene
  -> HOI opcional
  -> build_caption
  -> JSON
```

## Places365

Places365 predice la escena y se usa para:

- `core.scene.label`
- `core.scene.confidence`
- inferir `core.scene.indoor_outdoor`
- añadir vocabulario visual dependiente del grupo de escena

Rutas esperadas actualmente:

```text
/home/jovyan/projects/places365/categories_places365.txt
/home/jovyan/projects/models/places365/<architecture>_places365.pth.tar
```

## Detectores

### GroundingDINO

Usa un repo local importable como `groundingdino` y checkpoint:

```text
/home/jovyan/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
```

Ejemplo:

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --hoi none \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

### OWLv2

Usa `google/owlv2-base-patch16-ensemble` con `transformers`.

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector owlv2 \
  --hoi none \
  --output-dir outputs/workflow_b/owlv2_resnet50
```

OWLv2 convierte el prompt en candidate labels. `text_threshold` se conserva por compatibilidad de firma, pero se ignora porque OWLv2 no expone el mismo concepto de umbral textual por token.

## Vocabulario Actual

`src/workflow_b/vocabularies.py` produce:

- batches universales desde `UNIVERSAL_GROUNDING_PROMPT_BATCHES`
- batches extra según escena desde `SCENE_EXPANSION_VOCABS`
- batches AudioSet desde `resources/audioset_leaf_node_names.txt`, compartidos con Workflow A mediante `metrics/audioset_leaf_vocab.py`

Los términos son visuales: personas, animales, vehículos, estructuras, naturaleza, objetos, herramientas y comida. GroundingDINO/OWLv2 detectan evidencia visual, no audio.

`--vocab-mode audioset` usa términos detectables derivados de reglas hoja AudioSet. Los nodos marcados como `no`, los del bloque `FROM HERE` / `TO HERE`, los nodos `blacklist` de la ontología y las líneas sin evidencia estructurada se excluyen.

## Thresholds

- `box_threshold`: umbral de score para cajas del detector.
- `text_threshold`: relevante para GroundingDINO; ignorado por OWLv2.
- `min_confidence`: filtro posterior común en `filter_detections`.

En el código actual, los batches definen `box_threshold` y `text_threshold` propios; `min_confidence` sí se pasa directamente al postprocesado.

## HOI

Opciones:

- `--hoi none`: sin Human-Object Interaction.
- `--hoi dummy`: tripleta hardcoded para validar la fusión.
- `--hoi upt`: usa `UPTAdapter` con rutas externas. Es el default actual, por lo que para instalaciones nuevas conviene empezar con `--hoi none`.

## Output JSON

La salida tiene:

```json
{
  "core": {
    "image_id": "example",
    "scene": {},
    "entities": [],
    "observed_interactions": [],
    "spatial_relations": [],
    "environment": {},
    "caption": "..."
  },
  "extended": {
    "entities_extended": [],
    "global_geometry": {}
  },
  "metadata": {
    "detector": "owlv2",
    "scene_model": {},
    "hoi_backend": "none",
    "raw_hoi_count": 0,
    "box_threshold": 0.3,
    "text_threshold": 0.25,
    "min_confidence": 0.2
  }
}
```

Si `--output-dir` no se indica, el JSON se imprime en stdout. Si se indica, se guarda como `<image_stem>.json` o `--output-name`.
