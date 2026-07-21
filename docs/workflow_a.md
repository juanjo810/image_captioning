# Workflow A

Workflow A usa un VLM para producir una representación semántica estructurada desde una imagen. La salida principal es un JSON con sección `core`; opcionalmente puede incluir `acoustic_semantics.nodes` con nodos AudioSet inferidos solo desde evidencia visual.

## Estado

| Componente | Estado |
| --- | --- |
| Generación `core` con VLM | Implementado |
| Backend llama.cpp OpenAI-compatible | Implementado |
| Backend Gemma vía Hugging Face | Implementado |
| `--include-audioset-nodes` | Parcial |
| Promover AudioSet a `core.nodes` canónico | Planeado/TODO |

## CLI Real

Script:

```bash
python scripts/run_workflow_a.py --help
```

Argumentos implementados:

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--image` | ninguno | Ruta a una imagen individual. |
| `--image-dir` | ninguno | Directorio con imágenes `.jpg`. |
| `--limit` | `None` | Límite de imágenes al usar `--image-dir`. |
| `--model` | `llamacpp` | `gemma4` o `llamacpp`. |
| `--model-id` | depende del backend | Sobrescribe el id del modelo. |
| `--server-url` | `http://localhost:8889` | URL del servidor llama.cpp. |
| `--output-dir` | `outputs/workflow_a` | Directorio de salida. |
| `--max-new-tokens` | `2048` | Máximo de tokens nuevos. |
| `--temperature` | `0.0` | Temperatura de generación. |
| `--include-audioset-nodes` | off | Pide `acoustic_semantics.nodes`. |

## Backends Soportados

### llama.cpp

Backend por defecto. Usa `LlamaCppServerAdapter` y llama a:

```text
<server-url>/v1/chat/completions
```

La imagen se manda como data URL. Errores típicos: servidor apagado, URL incorrecta o respuesta no compatible con formato OpenAI.

### Gemma

`--model gemma4` usa `Gemma4Adapter` con `transformers`. El default es `google/gemma-4-E4B-it`. Requiere acceso al modelo y recursos suficientes.

## Imagen vs Directorio

Una imagen:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Directorio:

```bash
python scripts/run_workflow_a.py \
  --image-dir data/images \
  --limit 20 \
  --model llamacpp \
  --output-dir outputs/workflow_a
```

El modo directorio solo busca `*.jpg`.

## AudioSet Nodes

Con:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

El prompt añade una sección top-level `acoustic_semantics`:

```json
{
  "core": {},
  "acoustic_semantics": {
    "nodes": [
      {
        "id": "/m/...",
        "name": "AudioSet node name",
        "evidence": "visible evidence",
        "confidence": 0.7,
        "inference_type": "visible_source"
      }
    ]
  }
}
```

Los nodos permitidos se construyen desde `resources/audioset_leaf_node_names.txt` mediante `metrics/audioset_leaf_vocab.py` y se resuelven contra `ontology.json`.

Limitación importante: esto no es audio recognition. Son inferencias visuales conservadoras.

TODO: mover/promover estos nodos al esquema canónico planeado de `core.nodes`. Eso no está implementado.

## Output Folders

Workflow A crea:

```text
outputs/workflow_a/
  raw/                 Respuesta cruda del VLM.
  json/                JSON validado.
  captions/            Caption en `.txt`.
  manifests/manifest.jsonl
  failed/              Errores de parsing/validación.
```

El manifest de Workflow A es JSONL. Los scripts de evaluación semántica unificada esperan CSV; si vas a evaluar, prepara un CSV compatible o usa/ajusta el pipeline de manifests según el experimento.

## Errores Comunes

| Error | Causa probable | Solución |
| --- | --- | --- |
| `Could not connect to llama.cpp server` | Servidor apagado o URL incorrecta | Arranca llama.cpp y revisa `--server-url`. |
| JSON inválido | El VLM devolvió texto extra o campos fuera de schema | Baja temperatura, revisa `raw/`, aumenta `--max-new-tokens`. |
| Validación Pydantic falla | Campo faltante, categoría inválida o ids mal formados | Revisa `failed/<image>.json` y `raw/<image>.txt`. |
| No procesa directorio | Solo busca `.jpg` | Convierte/extiende inputs o usa `--image`. |
