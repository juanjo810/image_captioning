# AudioSet

AudioSet se usa en este repositorio como ontología acústico-semántica para organizar conceptos de sonido plausibles a partir de evidencia visual. No se usa como ground truth acústico real.

## Por Qué AudioSet

El proyecto busca describir imágenes de forma útil para tareas futuras de soundscape reasoning. AudioSet aporta:

- identificadores canónicos (`audioset_id`)
- nombres estables de nodos
- relaciones jerárquicas
- métricas a distintos niveles de abstracción

## Ontología DAG

AudioSet es una jerarquía tipo DAG, no un árbol estricto. Un nodo puede tener varios padres. Por eso las métricas no asumen una única ruta perfecta a la raíz.

La carga se implementa en:

```text
metrics/audioset_ontology.py
```

`AudioSetOntology` indexa:

- `node_by_id`
- `node_name_to_id`
- `children_map`
- `parents_map`
- `root_ids`
- `leaf_node_ids`

También expone utilidades para ancestors, top-level nodes, lowest common ancestor y tree distance.

## Blacklist y Nodos Abstractos

Cada nodo puede tener `restrictions`. El método `is_usable_label` excluye nodos con `blacklist`. Los nodos abstractos se excluyen de `leaf_node_ids`, pero pueden seguir participando en navegación jerárquica si existen en la ontología.

Valida la ontología con:

```bash
python scripts/evaluation/validate_audioset_ontology.py
```

## IDs Canónicos vs Prompt Terms

Hay dos niveles que no deben confundirse:

| Concepto | Ejemplo | Uso |
| --- | --- | --- |
| `prompt_terms` visuales | `car`, `river`, `person` | Términos para detectar evidencia visual. |
| `audioset_id` canónico | `/m/...` | Nodo real de AudioSet para métricas. |

Workflow B actualmente usa vocabulario visual para detectores. La integración AudioSet-aware está planeada, no implementada.

## Workflow A

Con `--include-audioset-nodes`, Workflow A pide al VLM una sección:

```json
{
  "acoustic_semantics": {
    "nodes": [
      {
        "id": "/m/...",
        "name": "Speech",
        "evidence": "visible evidence",
        "confidence": 0.7,
        "inference_type": "visible_source"
      }
    ]
  }
}
```

Los nodos permitidos se derivan desde `AUDIOSET_CONCEPT_TO_NODE_NAME` y se resuelven contra la ontología. El VLM debe escoger solo de esa lista.

Estado: parcial. La salida todavía no está promovida a `core.nodes`.

## Workflow B

Workflow B detecta objetos/escenas visuales. Luego las métricas pueden mapear esas etiquetas a nodos AudioSet mediante reglas deterministas en:

```text
metrics/audioset_semantics.py
```

Ejemplo conceptual:

```text
car -> Vehicle / Engine
river -> Water / Stream
bell tower -> Bell / Church bell
```

Esto es una proyección acústico-semántica, no una detección de sonido.

## Visual Genome Pseudo-References

Visual Genome proporciona objetos, relaciones y captions visuales. No proporciona etiquetas acústicas reales. Para evaluación AudioSet se generan **AudioSet pseudo-references** o **acoustic-semantic pseudo-references** proyectando objetos/relaciones de VG a nodos AudioSet.

## Limitaciones

- Una imagen de una calle no prueba que haya tráfico audible.
- Una persona visible no implica speech.
- Un instrumento visible puede sugerir música, pero no confirma sonido real.
- Las métricas AudioSet miden alineación semántica con pseudo-referencias, no precisión acústica real.
