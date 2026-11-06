# Troubleshooting

Problemas comunes y soluciones rápidas.

## `ModuleNotFoundError: No module named 'src.workflow_a'`

Estás ejecutando el script por ruta:

```bash
python scripts/run_workflow_a.py ...   # falla
```

Python pone `scripts/` en el path, no la raíz del repo. Ejecuta siempre como módulo, desde la raíz:

```bash
python -m scripts.run_workflow_a ...
```

Aplica a todos los scripts, incluidos los de `scripts/evaluation/`.

## llama.cpp server no está corriendo

Síntoma:

```text
Could not connect to llama.cpp server
```

Soluciones:

- arranca el servidor llama.cpp con soporte multimodal (modelo + `mmproj`)
- confirma que expone `/v1/chat/completions`
- revisa `--server-url` (default `http://localhost:8889`)

Si el servidor corre en **otro contenedor**, `localhost` apunta al contenedor equivocado: usa la IP del gateway del host, por ejemplo `--server-url http://172.17.0.1:8889`.

## No encuentro `--temperature` ni `--model` en Workflow A

No existen. El único backend es llama.cpp (`Gemma4Adapter` fue eliminado) y el adaptador envía solo `model`, `messages` y `max_tokens`. La temperatura, el top-p y el resto del sampling se configuran al lanzar `llama-server`; si la calidad del JSON es errática, ahí es donde hay que tocar.

`--model-id` sí existe, pero solo nombra el modelo en la petición y en `metadata`.

## JSON inválido del VLM

Síntomas: fallo de parsing, error Pydantic, fichero en `failed/`.

Revisa:

```text
outputs/workflow_a/raw/<image_id>.txt
outputs/workflow_a/failed/<image_id>.json
```

Prueba:

- subir `--max-new-tokens` (el JSON puede estar truncado)
- bajar la temperatura **en el servidor**
- usar un modelo multimodal más obediente al formato JSON
- para lotes, `run_workflow_a_with_retries` reintenta solo las imágenes fallidas

## `core.nodes` y/o `core.visual_terms` salen vacíos

No es un fallo: `caption` y `acoustic_caption` son campos separados, y `caption` depende solo de `visual_terms`, nunca de `nodes` — no puede haber una caption puntuable que mencione un sonido cuyo nodo se descartó en la validación (eso pasaba antes del *split*). Un `core.visual_terms` vacío significa que ningún término declarado tenía equivalente en el vocabulario cerrado de 210 términos; revisa `raw/` (y, en `five`, `metadata.n_free_visual_terms`/`n_mapped_visual_terms`) para confirmarlo.

## Falta GroundingDINO

Síntoma:

```text
ModuleNotFoundError: No module named 'groundingdino'
```

Solución: instala el repo GroundingDINO en el entorno `imagecap-b`, verifica que el import funciona y confirma las rutas que espera `scripts/run_grounding_dino_pipeline.py`.

## Faltan checkpoints

```text
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
/home/jovyan/projects/models/places365/resnet50_places365.pth.tar
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
```

UPT solo hace falta con `--legacy-visual-core --hoi upt`. El default es `--hoi none`, así que para probar el detector no necesitas UPT.

## Pocas o ninguna detección en Workflow B

Mira `metadata.n_raw_detections` y `metadata.n_filtered_detections` en el JSON de salida:

- muchas crudas y pocas filtradas → baja `--min-confidence` (default `0.30`) o sube `--nms-iou-threshold`
- pocas crudas → baja `--box-threshold`/`--text-threshold`; recuerda que, sin valor en el CLI, se usa el threshold de cada batch de vocabulario

## `--vocab-mode` parece no tener efecto

Es lo esperado sin `--legacy-visual-core`: la ruta por defecto fuerza `vocab_mode="audioset"` sea cual sea el flag. `metadata.vocab_mode` registra el valor efectivo (`"audioset"` en ese caso, o el flag real con `--legacy-visual-core`), no el flag crudo del CLI.

## OWLv2 ignora `text_threshold`

Esperado. `OWLv2Adapter.predict` conserva el parámetro por compatibilidad con `GroundingDINOAdapter`, pero OWLv2 filtra solo con `box_threshold`.

## Descargas de Hugging Face

OWLv2 y los modelos GGUF pueden requerir descarga. Si falla: comprueba conexión, permisos del modelo, `HF_HOME`/caché, y autentícate si el modelo lo exige.

## CUDA/Torch mismatch

Síntomas: errores de CUDA runtime, `torch.cuda.is_available()` falso, incompatibilidad entre `torch`, `torchvision` y el driver.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Instala versiones compatibles con tu driver. No mezcles los tres entornos: cada uno tiene su combinación verificada de Python/Torch/CUDA.

## Out of memory

Solo un proceso pesado de GPU a la vez: para `llama-server` antes de lanzar Workflow B, y al revés. En GPUs pequeñas, sirve los modelos GGUF en 8 bits y ajusta `-ngl` para decidir cuántas capas van a GPU.

## Java/SPICE

Las métricas de caption pueden requerir Java para SPICE. Si falla: instala Java, revisa permisos de descarga/caché, o reporta las métricas no-SPICE si el entorno no lo soporta.

## Referencias de Visual Genome no encontradas o vacías

Revisa `--vg-object-refs` y `--vg-relationship-refs`, que `image_id` coincida con el manifest, y que los CSV tengan las columnas `image_id`, `objects` y `relationships`.

Si el CSV de relaciones sale casi vacío, el problema suele ser la clave `name`/`names` del dump de VG. Ver [data_and_manifests.md](data_and_manifests.md).

## Métricas que dan 0.0 sin error

Si estás evaluando predicciones `AudioSetCoreJSON` con `evaluate_all_semantic_metrics.py` (o con `evaluate_structured_vg.py` / `evaluate_soundscape_semantics.py`), los ceros son artefactos: esos scripts leen `core.entities`, que el formato audioset no tiene, y no avisan. Usa `evaluate_audioset_semantics.py` y `evaluate_scene_awareness.py`.

## Manifest con columnas inconsistentes

`build_prediction_manifest.py` emite columnas distintas para cada formato, y `csv.DictWriter` toma las claves de la primera fila. No mezcles predicciones audioset y legacy en un mismo `--inputs`.

## Paths del manifest incorrectos

`json_path` se resuelve desde el directorio donde ejecutas el comando. Usa rutas absolutas o ejecuta desde la raíz del repo.

## El JSONL de Workflow A no sirve como manifest

Workflow A escribe `manifests/manifest.jsonl`, pero la evaluación espera CSV. Genera el CSV con `build_prediction_manifest.py` apuntando al directorio `json/`.

## ffmpeg/libopenh264

Si alguna dependencia multimedia falla por codecs, instala ffmpeg en el sistema o en el entorno. No es parte del flujo principal de JSON, pero aparece en entornos con paquetes de visión/vídeo.
