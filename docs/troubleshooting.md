# Troubleshooting

Problemas comunes y soluciones rápidas.

## llama.cpp Server No Está Corriendo

Síntoma:

```text
Could not connect to llama.cpp server
```

Soluciones:

- arranca el servidor llama.cpp con soporte multimodal
- confirma que expone `/v1/chat/completions`
- revisa `--server-url`

## URL Incorrecta

El default es:

```text
http://localhost:8889
```

Si tu servidor está en otro puerto:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --server-url http://localhost:<port>
```

## JSON Inválido del VLM

Síntomas:

- falla de parsing
- error Pydantic
- archivo en `failed/`

Revisa:

```text
outputs/workflow_a/raw/<image_id>.txt
outputs/workflow_a/failed/<image_id>.json
```

Prueba:

- `--temperature 0.0`
- aumentar `--max-new-tokens`
- usar un modelo multimodal más obediente al JSON

## Falta GroundingDINO

Síntoma:

```text
ModuleNotFoundError: No module named 'groundingdino'
```

Solución:

- instala el repo GroundingDINO en el entorno
- verifica que el import funcione
- confirma las rutas esperadas por `scripts/run_grounding_dino_pipeline.py`

## Falta Checkpoint

GroundingDINO espera:

```text
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
```

Places365 espera:

```text
/home/jovyan/projects/models/places365/resnet50_places365.pth.tar
```

UPT espera:

```text
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
```

Si solo quieres probar el detector sin HOI, usa `--hoi none`.

## Hugging Face Download Issues

OWLv2 y Gemma pueden descargar modelos. Si falla:

- comprueba conexión
- comprueba permisos del modelo
- revisa `HF_HOME` o la caché de Hugging Face
- autentica con Hugging Face si el modelo lo requiere

## CUDA/Torch Mismatch

Síntomas:

- errores de CUDA runtime
- `torch.cuda.is_available()` falso
- incompatibilidad entre `torch`, `torchvision` y drivers

Comprueba:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Instala versiones de torch/torchvision compatibles con tu driver.

## ffmpeg/libopenh264

Si alguna métrica o dependencia de multimedia falla con codecs, instala ffmpeg en el sistema o en el entorno. No es parte del flujo principal de JSON, pero puede aparecer en entornos con paquetes de visión/video.

## Java/SPICE

Las métricas de caption pueden requerir Java/SPICE según el script usado. Si falla SPICE:

- instala Java
- revisa permisos de descarga/cache
- considera reportar métricas no-SPICE si el entorno no lo soporta

## Visual Genome References No Encontradas

Síntomas:

- `FileNotFoundError`
- métricas vacías
- ids sin referencias

Revisa:

- `--vg-object-refs`
- `--vg-relationship-refs`
- que `image_id` coincida con el manifest
- que los CSV tengan columnas `image_id`, `objects` y `relationships`

## Paths del Manifest Incorrectos

`json_path` se resuelve desde el directorio donde ejecutas el comando. Usa rutas absolutas o ejecuta desde la raíz del repo.

## Workflow A Manifest JSONL vs CSV

Workflow A genera JSONL, pero evaluación semántica espera CSV. Convierte antes de llamar a `evaluate_all_semantic_metrics.py`.

## OWLv2 Ignora `text_threshold`

Esto es esperado. `OWLv2Adapter.predict` conserva `text_threshold` para compatibilidad con `GroundingDINOAdapter`, pero OWLv2 filtra con `box_threshold`.
