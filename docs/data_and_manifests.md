# Data and Manifests

Este repositorio no versiona datasets grandes, outputs generados ni checkpoints. La estructura de datos se prepara localmente.

## Layout Recomendado

```text
data/
  images/
    example.jpg
  vg/
    region_descriptions.json
    objects.json
    relationships.json
  aliases/
    object_alias.csv
    relationship_alias.csv
  vg_refs/
    vg_caption_references.csv
    vg_object_references.csv
    vg_relationship_references.csv

outputs/
  workflow_a/
  workflow_b/
  manifests/
  metrics/
```

## Imágenes

Ambos workflows, en modo directorio, buscan solo `*.jpg`:

- Workflow A: `--image-dir` + `--limit`
- Workflow B: `--image-dir` + `--limit` (el bucle es in-process; los modelos se cargan una sola vez)

## Outputs

### Workflow A

```text
outputs/workflow_a/
  raw/<image_id>.txt
  json/<image_id>.json
  captions/<image_id>.txt
  manifests/manifest.jsonl
  failed/<image_id>.json
```

Con `--legacy-visual-core`, todo eso cuelga de `outputs/workflow_a/legacy/`.

### Workflow B

Con `--output-dir`:

```text
outputs/workflow_b/<condition>/<image_id>.json
outputs/workflow_b/<condition>/legacy/<image_id>.json   # con --legacy-visual-core
```

Sin `--output-dir`, el JSON se imprime en stdout.

## Manifest CSV para evaluación

Los scripts de métricas leen un CSV con, como mínimo:

```csv
image_id,json_path,detector,scene_model,captioner
123,outputs/pilot_vg/visual_json_audioset/owlv2_resnet50/123.json,owlv2,resnet50,template
```

`json_path` apunta al JSON de predicción; las otras tres columnas agrupan el summary.

### `build_prediction_manifest.py`

```bash
python -m scripts.evaluation.build_prediction_manifest \
  --inputs outputs/pilot_vg/visual_json_audioset/owlv2_resnet50 \
           outputs/pilot_vg/visual_json_audioset/grounding_dino_resnet50 \
  --output outputs/manifests/predictions.csv
```

Infiere la configuración desde los nombres de directorio:

- `detector` / `scene_model`: del nombre del directorio padre, si empieza por `grounding_dino_` u `owlv2_`.
- Layout de Workflow A (`<output_dir>/json/` o `<output_dir>/legacy/json/`): el padre inmediato nunca identifica la ejecución, así que sube un nivel — `detector` queda como `unknown` y `scene_model` toma el nombre de `<output_dir>`.
- `captioner`: `gemma` si la ruta contiene `captions_gemma`, si no `template`.

Si usas otra convención de nombres, revisa el CSV resultante.

El script ramifica según el formato de cada predicción (`is_audioset_core_json`), y las columnas **no son las mismas**:

| Columnas comunes | Solo `AudioSetCoreJSON` | Solo `CoreJSON` legacy |
| --- | --- | --- |
| `image_id`, `json_path`, `detector`, `scene_model`, `captioner`, `caption`, `scene_label`, `scene_confidence`, `indoor_outdoor`, `total_object_coverage`, `object_density_proxy`, `metadata_detector`, `metadata_scene_model`, `metadata_caption_mode`, `metadata_call_mode`, `metadata_model_id` | `n_nodes`, `n_visual_terms` | `n_entities`, `n_interactions`, `crowd_level`, `activity_level` |

Las columnas de escena son las mismas en ambas ramas (`scene_label` / `scene_confidence` / `indoor_outdoor`).

`metadata_call_mode` solo es significativo en predicciones audioset de Workflow A (`single`, `three`, `two`, `five`); en cualquier otro caso sale vacío.

**No pases `--inputs` mezclando los dos formatos en una sola llamada**: las dos ramas devuelven columnas distintas y `csv.DictWriter` asume que las claves de la primera fila valen para todas. Genera un manifest por formato.

## Manifest JSONL de Workflow A

Workflow A escribe además su propio índice append-only, una línea por imagen:

```text
outputs/workflow_a/manifests/manifest.jsonl
```

```json
{
  "image_id": "example",
  "image_path": "data/images/example.jpg",
  "json_path": "outputs/workflow_a/json/example.json",
  "caption_path": "outputs/workflow_a/captions/example.txt"
}
```

Ese JSONL **no** es el CSV que esperan los scripts de evaluación. Para evaluar, genera el CSV con `build_prediction_manifest.py` apuntando al directorio `json/`.

## Referencias Visual Genome

```bash
python -m scripts.evaluation.build_vg_references \
  --manifest outputs/manifests/predictions.csv \
  --region-descriptions data/vg/region_descriptions.json \
  --objects data/vg/objects.json \
  --relationships data/vg/relationships.json \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --output-dir data/vg_refs
```

Genera:

```text
vg_caption_references.csv       image_id,reference_caption
vg_object_references.csv        image_id,objects
vg_relationship_references.csv  image_id,relationships
```

Trampa al preparar los datos de VG desde cero: el script lee solo la clave `names` en el subject/object de cada relación, pero el dump crudo de Visual Genome guarda `name` (singular) en la gran mayoría de los sujetos. Sobre un dump sin normalizar, `vg_relationship_references.csv` sale casi vacío y las métricas de relaciones dan ~0. El procedimiento de preparación del subconjunto descrito en [setup.md](setup.md) ya normaliza `name` → `names`, así que esto solo aplica si preparas los ficheros por tu cuenta saltándote ese paso.

Comprobarlo es inmediato: cuenta cuántas filas traen la columna `relationships` vacía. Que **algunas** salgan vacías es normal y no indica fallo de parseo — hay imágenes de Visual Genome sin relaciones anotadas. Lo que delata el problema de `name`/`names` es que salgan vacías *casi todas*.

## Alias maps

`vg_utils.load_alias_map` espera el formato `canonical,alias1,alias2,...`. Los ficheros `object_alias.txt` y `relationship_alias.txt` que distribuye Visual Genome ya lo cumplen.

## Errores comunes de paths

| Problema | Solución |
| --- | --- |
| `json_path` relativo resuelto desde otro directorio | Ejecuta desde la raíz del repo o usa rutas absolutas. |
| `image_id` no coincide entre manifest y referencias | Asegura que ambos usen el mismo stem/id. |
| CSV de relaciones vacío | Revisa el formato `subject::predicate::object` y la normalización `name`/`names`. |
| Manifest con columnas inconsistentes | Estabas mezclando predicciones audioset y legacy en un mismo `--inputs`. |
| JSONL de Workflow A usado como CSV | Genera el CSV con `build_prediction_manifest.py`. |
| Outputs/checkpoints en Git | Añádelos a `.gitignore`; no versionar generados pesados. |
