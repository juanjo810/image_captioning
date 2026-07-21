# Data and Manifests

Este repositorio no versiona datasets grandes, outputs generados ni checkpoints. La estructura de datos debe prepararse localmente.

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

Los paths son sugeridos. Los scripts aceptan rutas por argumento excepto varias rutas externas hardcoded en Workflow B.

## Imágenes

Workflow A en modo directorio busca solo:

```text
*.jpg
```

Workflow B procesa una imagen por ejecución con `--image`.

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

### Workflow B

Si se pasa `--output-dir`:

```text
outputs/workflow_b/<condition>/<image_id>.json
```

Si no se pasa `--output-dir`, el JSON se imprime en stdout.

## Manifest CSV para Evaluación

`evaluate_all_semantic_metrics.py` espera CSV. Columnas mínimas:

```csv
image_id,json_path,detector,scene_model,captioner
```

Columnas adicionales pueden existir, pero esas son las usadas por el evaluador.

Ejemplo:

```csv
image_id,json_path,detector,scene_model,captioner
123,outputs/workflow_b/owlv2_resnet50/123.json,owlv2,resnet50,template
```

`scripts/evaluation/build_prediction_manifest.py` puede construir un manifest desde directorios de JSON:

```bash
python scripts/evaluation/build_prediction_manifest.py \
  --inputs outputs/workflow_b/owlv2_resnet50 outputs/workflow_b/grounding_dino_resnet50 \
  --output outputs/manifests/predictions.csv
```

Nota: este helper infiere `detector`, `scene_model` y `captioner` desde nombres de directorio. Si usas otra convención, revisa el CSV resultante.

## Manifest JSONL de Workflow A

Workflow A escribe:

```text
outputs/workflow_a/manifests/manifest.jsonl
```

con registros que incluyen:

```json
{
  "image_id": "example",
  "image_path": "data/images/example.jpg",
  "json_path": "outputs/workflow_a/json/example.json",
  "caption_path": "outputs/workflow_a/captions/example.txt"
}
```

Ese JSONL no es directamente el CSV esperado por `evaluate_all_semantic_metrics.py`.

## Visual Genome References

Se pueden construir con:

```bash
python scripts/evaluation/build_vg_references.py \
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
vg_caption_references.csv
vg_object_references.csv
vg_relationship_references.csv
```

## Errores Comunes de Paths

| Problema | Solución |
| --- | --- |
| `json_path` relativo desde otro directorio | Ejecuta desde la raíz del repo o usa rutas absolutas. |
| `image_id` no coincide entre manifest y refs | Asegura que ambos usen el mismo stem/id. |
| CSV de relaciones vacío | Revisa formato `subject::predicate::object`. |
| Workflow A JSONL usado como CSV | Convierte a CSV compatible antes de evaluar. |
| Outputs/checkpoints en Git | Añade a `.gitignore`; no versionar archivos pesados generados. |
