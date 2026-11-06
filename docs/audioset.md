# AudioSet

AudioSet se usa en este repositorio como ontología acústico-semántica para organizar conceptos de sonido plausibles a partir de evidencia visual. No se usa como ground truth acústico real.

## Por Qué AudioSet

El proyecto busca describir imágenes de forma útil para tareas futuras de soundscape reasoning. AudioSet aporta:

- identificadores canónicos (`audioset_id`)
- nombres estables de nodos
- relaciones jerárquicas
- métricas a distintos niveles de abstracción

## Ontología DAG

AudioSet es una jerarquía tipo DAG, no un árbol estricto: un nodo puede tener varios padres. Por eso las métricas no asumen una única ruta a la raíz.

La carga se implementa en `metrics/audioset_ontology.py`. `AudioSetOntology` indexa:

- `node_by_id`
- `node_name_to_id`
- `children_map`
- `parents_map`
- `root_ids`
- `leaf_node_ids`

También expone utilidades para ancestors, top-level nodes (`top_levels`), padres o el propio nodo (`parents_or_self`), lowest common ancestor y tree distance.

## Blacklist y nodos abstractos

Cada nodo puede tener `restrictions`. `is_usable_label` excluye los nodos con `blacklist`. Los nodos abstractos quedan fuera de `leaf_node_ids`, pero siguen participando en la navegación jerárquica.

Valida la ontología con:

```bash
python -m scripts.evaluation.validate_audioset_ontology
```

## Vocabulario hoja compartido

`resources/audioset_leaf_node_names.txt` define el conjunto hoja del proyecto. `metrics/audioset_leaf_vocab.py` lo parsea y lo resuelve contra `ontology.json`, y expone:

- `allowed_audioset_leaf_nodes()` — los nodos AudioSet permitidos
- `audioset_detectable_terms()` — los 210 términos visuales detectables
- `audioset_leaf_rules()` — las reglas que ligan términos visuales a nodos hoja

Ese mismo vocabulario se usa en tres sitios, y esa es justamente la propiedad que se quiere mantener:

| Consumidor | Qué toma |
| --- | --- |
| Workflow A | `src/workflow_a/audioset_nodes.py` — allow-list de nodos y de `visual_terms` para el prompt |
| Workflow B | 7 batches de prompts de detección de hasta 30 términos, y las reglas hoja de la proyección |
| Evaluación | las mismas reglas hoja, para derivar referencias desde Visual Genome |

Se excluyen los nodos marcados como `no`, los del bloque `FROM HERE` / `TO HERE`, los nodos `blacklist` de la ontología y las líneas sin evidencia estructurada.

## IDs canónicos vs términos visuales

Dos niveles que no deben confundirse:

| Concepto | Ejemplo | Uso |
| --- | --- | --- |
| Término visual | `car`, `river`, `person` | Lo que se detecta o se declara como evidencia visual. |
| `audioset_id` canónico | `/m/...` | Nodo real de AudioSet, lo que se puntúa. |

La ligadura entre ambos es `visual_evidence_terms` en cada nodo: qué objetos concretos justifican ese sonido.

## Workflow A

Por defecto, Workflow A emite `core.nodes` directamente como `AudioSetCoreNode`, escogidos de la allow-list. Cada nodo declara `audioset_id`, `audioset_name`, `node_type` (`visible_source` o `visible_action`), `evidence` y `visual_evidence_terms`. Workflow A **no** rellena `parent_ids`, `top_level_ids` ni `confidence` de nodo.

Con `--legacy-visual-core --include-audioset-nodes`, en cambio, los nodos van al bloque legacy `acoustic_semantics.nodes`, con `inference_type` de cuatro valores (`visible_source`, `visible_action`, `scene_affordance`, `uncertain`).

Detalles en [workflow_a.md](workflow_a.md).

## Workflow B

Workflow B detecta objetos visuales y los proyecta sobre nodos AudioSet mediante las reglas hoja (`src/workflow_b/audioset_projection.py`). Es la única de las dos rutas que rellena `parent_ids` y `top_level_ids`, y siempre rellena `visual_evidence_terms` y `confidence`.

Ejemplo conceptual:

```text
car        -> Motor vehicle (road)
river      -> Stream
bell tower -> Church bell
```

Esto es una proyección acústico-semántica, no detección de sonido.

`metrics/audioset_semantics.py` mantiene además `AUDIOSET_CONCEPT_TO_NODE_NAME`, el mapeo de conceptos visuales/VG a nombres de nodo que se usa en el eje "qué sonidos ocurrirían plausiblemente en esta escena visualmente fundamentada".

## Predicciones vs referencias en evaluación

`predicted_audioset_tag_counts` tiene hoy `pred_source="core_nodes"` por defecto: lee `core.nodes[].audioset_id` directamente, sin re-derivar nada. Las opciones `core_rules`, `vlm_nodes` y `union` se conservan como legacy.

`reference_audioset_tag_counts` deriva las referencias exclusivamente desde objetos y relaciones de Visual Genome, con las mismas reglas hoja.

## Visual Genome pseudo-references

Visual Genome aporta objetos, relaciones y descripciones visuales. No aporta etiquetas acústicas. Para evaluación AudioSet se generan **AudioSet pseudo-references** (o *acoustic-semantic pseudo-references*) proyectando objetos/relaciones de VG sobre nodos AudioSet.

## Limitaciones

- Una imagen de una calle no prueba que haya tráfico audible.
- Una persona visible no implica speech.
- Un instrumento visible puede sugerir música, pero no confirma sonido real.
- Las métricas AudioSet miden alineación semántica con pseudo-referencias, no precisión acústica.
