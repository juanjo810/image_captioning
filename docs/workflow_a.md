# Workflow A

Workflow A usa un VLM para producir la representación semántica estructurada directamente desde la imagen. No hay detector en el bucle.

Hay dos rutas de salida, elegidas con `--legacy-visual-core`:

- **por defecto (audioset core)**: `AudioSetCoreJSON` — `image_id`, `scene`, `nodes[]`, `caption`. Sin `entities`, `observed_interactions` ni `spatial_relations`.
- **`--legacy-visual-core`**: `CoreJSON` clásico, con `acoustic_semantics.nodes` opcional si además se pasa `--include-audioset-nodes`.

## Estado

| Componente | Estado |
| --- | --- |
| Generación `AudioSetCoreJSON` | Implementado (ruta por defecto) |
| `--call-mode single` / `three` / `two` | Implementado |
| Generación `CoreJSON` legacy | Implementado (`--legacy-visual-core`) |
| `acoustic_semantics.nodes` legacy | Implementado (`--include-audioset-nodes`, solo con `--legacy-visual-core`) |
| Backend llama.cpp OpenAI-compatible | Único backend |
| Backend Gemma vía Hugging Face | **Eliminado** (`Gemma4Adapter` borrado) |
| Reintentos automáticos en lote | Implementado (`run_workflow_a_with_retries.py`) |

## CLI Real

```bash
python -m scripts.run_workflow_a --help
```

Ejecuta siempre como módulo desde la raíz del repo. `python scripts/run_workflow_a.py` falla con `ModuleNotFoundError: No module named 'src.workflow_a'`.

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--image` | ninguno | Ruta a una imagen individual. |
| `--image-dir` | ninguno | Directorio con imágenes `.jpg`. |
| `--limit` | `None` | Límite de imágenes al usar `--image-dir`. |
| `--model-id` | `local-vlm` | Id de modelo que se manda al servidor y se registra en `metadata`. |
| `--server-url` | `http://localhost:8889` | URL del servidor llama.cpp. |
| `--output-dir` | `outputs/workflow_a` | Directorio de salida. |
| `--max-new-tokens` | `2048` | Se envía como `max_tokens` al servidor. |
| `--include-audioset-nodes` | off | Pide `acoustic_semantics.nodes`. Solo tiene efecto con `--legacy-visual-core`. |
| `--legacy-visual-core` | off | Usa el `CoreJSON` clásico en lugar del audioset core. |
| `--call-mode` | `single` | `single`, `three` o `two`. Solo tiene sentido sin `--legacy-visual-core`. |

**No existen `--model` ni `--temperature`.** El único backend es llama.cpp, y el adaptador envía únicamente `model`, `messages` y `max_tokens`: temperatura, top-p y demás sampling se configuran al lanzar `llama-server`, no desde este CLI.

### Runner con reintentos

```bash
python -m scripts.run_workflow_a_with_retries --help
```

Acepta los mismos argumentos, más:

| Argumento | Default | Descripción |
| --- | --- | --- |
| `--max-retries` | `2` | Reintentos extra por imagen fallida (3 intentos totales). |
| `--retry-delay` | `0.0` | Segundos de espera entre rondas de reintento. |

Reintenta solo las imágenes que fallaron parsing/validación y escribe un `retry_report.json` (listas de éxitos y fallos) en `--output-dir`. Pensado para lotes desatendidos, para no repetir toda la tanda por unos pocos fallos transitorios del VLM.

## Ruta por defecto: audioset core

### `--call-mode single` (por defecto)

`prompt_builder.build_workflow_a_audioset_core_prompt` construye el JSON en **dos etapas dentro de una sola respuesta**:

1. Etapa 1: una lista `visual_terms`, escogida solo del vocabulario compartido de términos AudioSet-detectables (`audioset_nodes.default_allowed_visual_terms`), el mismo que usan los detectores de Workflow B.
2. Etapa 2: los `nodes` del `AudioSetCoreJSON`, cada uno derivado de los términos declarados en la etapa 1 (y con `visual_evidence_terms` como subconjunto de ellos).

Es **una sola llamada HTTP**, no dos: el orden aprovecha el condicionamiento autorregresivo (el modelo no puede contradecir lo que ya escribió en la etapa 1) en lugar de una petición extra.

### `--call-mode three`

Tres llamadas independientes, con su propio prompt cada una:

1. `build_workflow_a_audioset_core_scene_prompt` — escena + `visual_terms`.
2. `build_workflow_a_audioset_core_nodes_prompt` — nodos, recibiendo los `visual_terms` anteriores **como texto**, no por condicionamiento autorregresivo.
3. `build_workflow_a_audioset_core_caption_prompt` — caption, recibiendo escena/`visual_terms`/nodos ya decididos.

La etapa de caption es deliberadamente **text-only** (`image_path=None`): todo lo que necesita ya está decidido, y adjuntar la imagen solo invitaría al modelo a describir lo que vuelve a ver en vez de resumir estrictamente los datos dados.

El intercambio frente a `single` es explícito: se pierde el condicionamiento autorregresivo que mantiene coherente una etapa con la anterior, a cambio de que cada etapa reciba toda la atención y el presupuesto del modelo en una tarea más estrecha. Se mantiene para experimentación y comparación, no como sustituto del default.

### `--call-mode two`

Dos llamadas, separando solo la etiqueta de escena:

1. `build_workflow_a_audioset_core_free_scene_prompt` (con imagen): pide `visual_terms`, `nodes` y `caption` junto con una descripción de escena **libre**, en palabras del modelo, sin la allow-list de Places365.
2. `build_workflow_a_scene_mapping_prompt` (text-only): mapea esa descripción libre más los `visual_terms` ya declarados sobre la allow-list oficial de Places365.

La caption se escribe en la fase 1, a partir de la descripción **libre**, no de la etiqueta oficial: la fase 2 solo sobrescribe `core.scene`. El fichero de `raw/` incluye además la escena libre ya parseada bajo su propia cabecera `=== PHASE 1 SCENE (free-form, parsed) ===`, para poder compararla con la etiqueta mapeada sin bucear en el JSON entero de la fase 1.

### Escena y tipos de nodo

En `single` y `three` la escena se pide directamente como par `label` + `indoor_outdoor` desde la allow-list `places365_mapping.allowed_places365_labels()`; en `two` solo llega a esa forma tras la fase 2. **Ningún modo pide `confidence` de escena**: los valores que produce el modelo son inventados, no medidos.

`node_type` solo admite `visible_source` o `visible_action`. `scene_affordance` y `uncertain` se eliminaron de esta ruta: `scene_affordance` era el único tipo al que el prompt permitía saltarse `visual_evidence_terms`, que era a la vez el agujero real del validador y una asimetría con Workflow B, cuyos nodos están evidenciados por construcción. (El `inference_type` del bloque legacy `acoustic_semantics` sigue teniendo los cuatro valores.)

### Validación

`validator.normalize_audioset_core_payload` hace, en orden:

- quita la clave de trabajo `visual_terms` (no forma parte de `AudioSetCoreJSON`);
- valida `scene.label` contra la allow-list de Places365, derivando `indoor_outdoor` con `infer_indoor_outdoor_from_scene` si el modelo lo omite o lo estropea, en vez de fiarse de un valor inventado;
- exige que `audioset_id` y `audioset_name` de cada nodo vengan de la **misma entrada** de la allow-list AudioSet;
- filtra `visual_evidence_terms` de cada nodo al vocabulario de términos detectables y, si el payload trae una lista `visual_terms` no vacía (los tres modos la rellenan), también al conjunto de términos realmente declarados, para que un nodo no pueda citar un objeto que el modelo nunca dijo ver;
- **descarta el nodo entero** si se queda sin `visual_evidence_terms` tras ese filtrado.

Consecuencia conocida y no resuelta: eso puede dejar `core.nodes` vacío en una imagen cuya caption —escrita antes de validar, en la misma llamada— sí menciona un sonido.

En esta ruta los nodos AudioSet se piden y validan **siempre**: `--include-audioset-nodes` no hace nada aquí.

### Ejemplo de salida

```json
{
  "core": {
    "image_id": "example",
    "scene": {
      "label": "street",
      "indoor_outdoor": "outdoor"
    },
    "nodes": [
      {
        "node_id": "n1",
        "audioset_id": "/m/012f08",
        "audioset_name": "Motor vehicle (road)",
        "node_type": "visible_source",
        "evidence": "several cars driving along the road",
        "visual_evidence_terms": ["car"]
      }
    ],
    "caption": "..."
  },
  "metadata": {
    "workflow": "A",
    "model_id": "local-vlm",
    "adapter": "LlamaCppServerAdapter",
    "prompt_version": "workflow_a_audioset_core_v1",
    "call_mode": "single",
    "image_id": "example",
    "include_audioset_nodes": true,
    "use_legacy_core": false,
    "generation_params": { "max_new_tokens": 2048 }
  }
}
```

`prompt_version` en esta ruta es `workflow_a_audioset_core_v1`, `_three_call_v1` o `_two_call_v1` según el `--call-mode`.

El JSON se serializa con `exclude_none=True`: Workflow A nunca rellena `parent_ids`, `top_level_ids`, `confidence` de nodo ni `confidence` de escena, y esos campos se omiten en vez de emitirse como `null`. Workflow A **no emite sección `extended`**.

## Ruta legacy: `--legacy-visual-core`

`prompt_builder.build_workflow_a_legacy_visual_prompt` pide el `CoreJSON` clásico y `validator.validate_legacy_workflow_a_output` lo revalida. Con `--include-audioset-nodes` se pide además un bloque hermano `acoustic_semantics`, comprobado por `validator.validate_legacy_acoustic_semantics_output` contra la misma allow-list AudioSet:

```json
{
  "core": { "...": "CoreJSON clásico" },
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

`prompt_version` aquí es `workflow_a_legacy_visual_core_v1` o `workflow_a_legacy_visual_core_audioset_v1`.

Los nodos permitidos se construyen desde `resources/audioset_leaf_node_names.txt` mediante `metrics/audioset_leaf_vocab.py` y se resuelven contra `ontology.json`.

Limitación importante en ambas rutas: esto no es audio recognition. Son inferencias visuales conservadoras.

## Backend

Único backend: `LlamaCppServerAdapter`, que llama a

```text
<server-url>/v1/chat/completions
```

La imagen se manda como data URL. `BaseVLM.generate` y `LlamaCppServerAdapter.generate` aceptan `image_path=None` para llamadas text-only (las usan la etapa de caption de `three` y la fase 2 de `two`). Errores típicos: servidor apagado, URL incorrecta, o respuesta no compatible con el formato OpenAI.

Si el servidor corre en otro contenedor Docker, no uses `localhost`: alcánzalo por la IP del gateway del host (por ejemplo `--server-url http://172.17.0.1:8889`).

## Imagen vs Directorio

```bash
python -m scripts.run_workflow_a \
  --image data/images/example.jpg \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

```bash
python -m scripts.run_workflow_a \
  --image-dir data/images \
  --limit 20 \
  --output-dir outputs/workflow_a
```

El modo directorio solo busca `*.jpg`.

## Output Folders

```text
outputs/workflow_a/
  raw/                 Respuesta cruda del VLM.
  json/                JSON validado.
  captions/            Caption en `.txt`.
  manifests/manifest.jsonl
  failed/              Errores de parsing/validación.
```

Con `--legacy-visual-core` todo eso cuelga de un subdirectorio extra `legacy/`:

```text
outputs/workflow_a/legacy/json/<image_id>.json
```

En `raw/`, los modos `three` y `two` concatenan todas las etapas/fases bajo cabeceras `=== STAGE n ===` / `=== PHASE n ===`.

Cuando falla la validación, el pipeline escribe en `failed/` en vez de fallar en silencio; la excepción **sí** se propaga al llamante (por eso existe el runner con reintentos).

El manifest de Workflow A es JSONL con `image_id`, `image_path`, `json_path` y `caption_path`. Los scripts de evaluación esperan un CSV: constrúyelo con `scripts/evaluation/build_prediction_manifest.py` apuntando al directorio `json/`.

## Errores Comunes

| Error | Causa probable | Solución |
| --- | --- | --- |
| `Could not connect to llama.cpp server` | Servidor apagado, URL incorrecta o `localhost` entre contenedores | Arranca `llama-server` y revisa `--server-url`. |
| JSON inválido | El VLM devolvió texto extra o campos fuera de schema | Revisa `raw/`, sube `--max-new-tokens`, baja la temperatura **en el servidor**. |
| Validación Pydantic falla | Campo faltante, etiqueta de escena fuera de la allow-list, `audioset_id`/`audioset_name` descasados | Revisa `failed/<image>.json` y `raw/<image>.txt`. |
| `core.nodes` vacío pero la caption menciona sonidos | Todos los nodos perdieron su evidencia en la validación | Comportamiento conocido; revisa los `visual_terms` declarados en `raw/`. |
| No procesa el directorio | Solo busca `.jpg` | Convierte los inputs o usa `--image`. |
| Poca diversidad de etiquetas de escena | La allow-list completa junto a la imagen sesga al modelo | Prueba `--call-mode two`. |
