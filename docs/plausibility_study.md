# Estudio humano de plausibilidad

La evaluación automática compara las predicciones con pseudo-referencias derivadas de Visual Genome mediante las mismas reglas que usan los sistemas. Mide acuerdo con esas reglas, no si un sonido es plausible en la imagen. Este estudio aporta una referencia independiente: personas que juzgan, sin saber qué sistema propuso qué, si cada sonido se oiría en la escena y si cada objeto citado como evidencia se ve de verdad.

Este documento tiene dos partes: el protocolo para quien coordina el estudio y la guía para quien anota.

## Parte 1. Protocolo (coordinación)

### Qué se anota

Para cada imagen se construye un *pool* con:

- todos los nodos AudioSet predichos por cualquiera de los sistemas comparados;
- los nodos de la pseudo-referencia de esa imagen;
- dos sonidos al azar del vocabulario del estudio, que no proponía nadie para esa imagen. No son una trampa: dan la tasa de plausibilidad de un sonido cualquiera, que sirve de línea base;
- los términos visuales que algún sistema citó como evidencia, más un término al azar.

Cada anotador responde dos preguntas por imagen:

1. **Objetos:** ¿se ve en la imagen? (Sí / No / Dudoso).
2. **Sonidos:** en una grabación de unos 10 segundos hecha en el lugar y el momento de la foto, ¿se oiría? (Sí / Posible / Improbable / No). Con «Improbable» o «No» se pide el motivo: no hay fuente, fuente inactiva, no encaja con el lugar, u otro.

### Preparar el paquete

```bash
python -m scripts.evaluation.build_plausibility_pool \
  --manifest outputs/manifests/3000/predictions_manifest_3000.csv \
  --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \
  --image-ids outputs/seeds_500/image_ids.txt \
  --image-dir ../data/subsets/vg_5000_images \
  --n-images 300 --n-pilot 30 --seed 0 \
  --output-dir outputs/plausibility_study
```

Genera en `outputs/plausibility_study/`:

| Fichero | Para quién | Qué es |
| --- | --- | --- |
| `images/` | anotadores | Copia de las imágenes seleccionadas. |
| `annotate_pilot.html` | anotadores | Página de anotación del piloto (30 imágenes distintas de las del estudio). |
| `annotate_main.html` | anotadores | Página de anotación del estudio (300 imágenes). |
| `study_*.json` | nadie en particular | Los ítems que lleva dentro cada página. No contienen nombres de sistema. |
| `key_*.json` | **solo coordinación** | Qué sistema predijo cada nodo y con qué evidencia; qué nodos son de referencia o aleatorios. Hace falta para el análisis. **No se entrega a los anotadores.** |

La selección de imágenes usa una semilla fija y conserva la proporción de imágenes con pseudo-referencia vacía. Las imágenes del piloto no se repiten en el estudio.

### Repartir

A cada anotador se le entrega una carpeta con `annotate_pilot.html`, `annotate_main.html` y la carpeta `images/` al lado, tal como salen. La página se abre con doble clic en un navegador de escritorio (Chrome, Edge o Firefox); no necesita instalación ni conexión.

Cada anotador recibe un código (por ejemplo `A1`, `A2`, `A3`) y lo usa siempre. La página baraja el orden de las imágenes y de los ítems con ese código, de modo que cada anotador ve un orden distinto y reproducible.

### Fases

1. **Piloto.** Los anotadores hacen las 30 imágenes del piloto y envían su CSV. Se revisan juntos los desacuerdos y se ajusta esta guía si hace falta. Las respuestas del piloto no entran en los resultados.
2. **Estudio.** Cada anotador hace las 300 imágenes por su cuenta, sin comentar casos con los demás, y envía su CSV.
3. **Análisis.** Acuerdo entre anotadores (alfa de Krippendorff), decisión por mayoría y métricas por sistema.

### Qué devuelve cada anotador

El botón «Descargar resultados» genera `plausibility_v1_main_<código>.csv`, con una fila por respuesta:

| Columna | Contenido |
| --- | --- |
| `study_id`, `annotator` | Identificador del estudio y código del anotador. |
| `position` | Posición en que ese anotador vio la imagen. |
| `image_id` | Imagen. |
| `kind` | `object` o `sound`. |
| `item_id`, `item_name` | Término visual, o identificador y nombre del nodo AudioSet. |
| `answer` | Objetos: `yes`, `no`, `unsure`. Sonidos: `1` Sí, `2` Posible, `3` Improbable, `4` No. |
| `reason` | Solo con `3` o `4`: `absent`, `inactive`, `place`, `other`. |
| `image_seconds`, `image_done` | Tiempo acumulado en la imagen y si quedó completa. |

### Analizar

Los CSV que devuelven los anotadores se guardan en una carpeta (por ejemplo `outputs/plausibility_study/results/`) y se analizan con la clave:

```bash
python -m scripts.evaluation.analyze_plausibility \
  --annotations outputs/plausibility_study/results/plausibility_v1_pilot_*.csv \
  --key outputs/plausibility_study/key_pilot.json \
  --output-dir outputs/plausibility_study/analysis_pilot
```

Para el estudio se usan `plausibility_v1_main_*.csv` y `key_main.json`, y se añade `--per-image outputs/final_metrics/stats_3000/audioset_per_image.csv` para relacionar la puntuación automática con la humana.

Cómo decide el análisis:

- Solo cuentan las imágenes que el anotador dejó completas.
- Una imagen entra en el consenso cuando la completaron al menos `--min-annotators` anotadores (2 por defecto; con `1` se pueden mirar los resultados de un solo anotador).
- El consenso de cada ítem es la mayoría estricta de quienes lo valoraron. Un empate cuenta como no plausible o no presente; el resumen dice cuántos hubo.
- «Plausible» se lee de dos maneras y se calculan las dos: amplia («Sí» o «Posible») y estricta (solo «Sí»). La amplia es la principal, porque la tarea es proponer sonidos plausibles, no sonidos seguros.
- Si un sistema no tiene predicción para una imagen (ejecución fallida), esa imagen no cuenta para él. Las comparaciones entre dos sistemas usan las imágenes en que ambos tienen predicción.

Escribe en `--output-dir`:

| Fichero | Contenido |
| --- | --- |
| `summary.md` | Todas las tablas. |
| `annotators.csv` | Imágenes completadas, tiempo y distribución de respuestas de cada anotador. |
| `agreement.csv` | Alfa de Krippendorff con su intervalo: escala de cuatro niveles (ordinal), sus dos lecturas binarias y presencia de objetos. Con tres o más anotadores, también por pares. |
| `items_consensus.csv` | Una fila por ítem, con la respuesta de cada anotador y el consenso. |
| `systems.csv` | Métricas por sistema con intervalos. Incluye `pseudo_reference` (la pseudo-referencia puntuada como un sistema más) y `random_baseline` (los sonidos al azar). |
| `comparisons.csv` | Diferencias pareadas entre sistemas. Por defecto, las mismas comparaciones que la evaluación automática; `--compare A:B` las sustituye. |
| `reasons.csv` | Motivos de los sonidos predichos que se juzgaron no plausibles. |
| `nodes.csv` | Por nodo AudioSet: en cuántas imágenes estuvo en el pool y cuántas veces se juzgó plausible. |
| `disagreements.csv` | Ítems en que los anotadores no coinciden. Es lo que se revisa tras el piloto. |
| `automatic_vs_human.csv` | Solo con `--per-image`: F1 contra pseudo-referencia y F1 contra el consenso humano por sistema, y su correlación por imagen. |

### Criterio para pasar del piloto al estudio

Se fija antes de ver los resultados, sobre el alfa de la lectura amplia de los sonidos, que es la principal:

| Alfa | Qué se hace |
| --- | --- |
| 0,67 o más | Se pasa al estudio. La guía solo se toca para aclarar casos concretos. |
| Entre 0,50 y 0,67 | Se revisan los desacuerdos con los anotadores, se aclara la guía y se repite el piloto con imágenes nuevas. |
| Menos de 0,50 | La pregunta o la escala no funcionan. Se rediseñan antes de seguir. |

Krippendorff propone 0,80 como umbral de fiabilidad y 0,667 como mínimo para sacar conclusiones tentativas. Con 30 imágenes el intervalo del alfa es ancho, así que el valor se lee junto a su intervalo. El alfa por pares sirve para detectar a un anotador que usa la escala de otra manera que el resto.

### Qué se reporta

- F1 contra el consenso humano: media por imagen del F1 entre los nodos predichos y los sonidos del pool juzgados plausibles, con la misma convención que la métrica automática (predicción y referencia vacías cuentan 1).
- Precisión validada por humanos de cada sistema: proporción de sus nodos juzgados plausibles, con la lectura amplia y con la estricta.
- Cobertura: proporción de los sonidos plausibles del pool que cada sistema recupera. El pool solo contiene lo que propuso algún sistema o la pseudo-referencia, así que es una cota superior de la cobertura real y hay que decirlo.
- Abstención: en las imágenes donde ningún sonido del pool se juzgó plausible, proporción en que el sistema no predice nada.
- Fidelidad de la evidencia: proporción de nodos cuyos términos citados se ven en la imagen.
- Origen de los errores: cruce entre evidencia vista y sonido plausible. Separa los errores de percepción (el objeto citado no está en la imagen) de las reglas que sobregeneran (el objeto está, pero el sonido no es plausible).
- Validez de la pseudo-referencia: proporción de sus nodos juzgados plausibles, y correlación por imagen entre el F1 contra pseudo-referencia y el F1 contra el juicio humano.
- Motivos de implausibilidad por sistema y por nodo (fuente ausente, inactiva, lugar).
- Línea base: tasa de plausibilidad de los sonidos aleatorios.

Los intervalos de confianza se calculan remuestreando imágenes, no juicios, porque los nodos de una misma imagen no son independientes.

## Parte 2. Guía para anotadores

Gracias por participar. Vas a ver fotografías y, para cada una, una lista de objetos y una lista de sonidos. No hay respuestas correctas ni preguntas trampa: interesa tu juicio.

### Antes de empezar

- Usa un ordenador, no el móvil, con el navegador Chrome, Edge o Firefox.
- Abre el fichero `annotate_pilot.html` (o `annotate_main.html`) con doble clic. La carpeta `images` tiene que estar junto al fichero.
- Escribe el código que te han dado y úsalo siempre.
- Puedes dejarlo y seguir otro día en el mismo navegador: el progreso se guarda solo. Al terminar cada sesión pulsa «Copia de seguridad» y conserva el fichero, por si acaso.
- Reserva sesiones de no más de una hora. Si te cansas, para.

### Tarea 1: objetos

Para cada objeto de la lista, indica si **se ve** en la imagen.

- **Sí:** se ve, aunque sea pequeño o esté parcialmente tapado.
- **No:** no aparece.
- **Dudoso:** no se distingue bien, o no sabes si lo que ves cuenta como ese objeto.

No importa si el objeto es relevante o no; solo si está.

### Tarea 2: sonidos

Imagina que alguien graba unos 10 segundos de audio **en el lugar y en el momento de la foto**. Para cada sonido de la lista, ¿se oiría en esa grabación?

| Respuesta | Cuándo |
| --- | --- |
| **Sí** | Casi seguro que se oiría. La fuente está en la imagen y está sonando o haciendo lo que produce ese sonido. |
| **Posible** | Podría oírse, pero la imagen no lo asegura. La fuente está y podría estar sonando. |
| **Improbable** | Hay algo relacionado, pero no parece que esté sonando en ese momento. |
| **No** | Nada en la imagen puede producirlo, o no encaja con el lugar. |

Si eliges «Improbable» o «No», marca el motivo:

- **No hay fuente:** lo que produciría el sonido no aparece.
- **Fuente inactiva:** aparece, pero no está haciendo lo que suena.
- **No encaja con el lugar:** el sonido no corresponde a ese sitio o situación.
- **Otro:** cualquier otro motivo.

### Criterios

- **Juzga esta foto, no lo que suele pasar.** Que en las calles suela haber tráfico no basta: mira si en esta calle hay coches circulando.
- **Lo que cuenta es el momento de la foto.** Un coche aparcado no hace ruido de motor. Una persona con la boca cerrada que mira a cámara no está hablando.
- **La fuente puede estar fuera de plano si la escena lo deja claro.** En una foto tomada desde dentro de un tren en marcha, el ruido del tren se oiría aunque no se vea el tren entero.
- **Los sonidos son específicos.** «Car alarm» es la alarma de un coche, no cualquier sonido de coche. Lee la descripción que hay debajo del nombre.
- **Si dudas entre dos respuestas, elige la del medio que mejor refleje tu duda.** «Posible» e «Improbable» están para eso.
- **No intentes adivinar qué se espera.** Algunos sonidos de la lista no tienen nada que ver con la imagen; es normal responder «No» muchas veces.

### Ejemplos

| En la imagen | Sonido | Respuesta razonable |
| --- | --- | --- |
| Un perro ladrando a otro, con la boca abierta | Bark | Sí |
| Un perro tumbado, dormido | Bark | Improbable, fuente inactiva |
| Dos personas conversando en una mesa | Male speech, man speaking | Sí, si al menos una es un hombre |
| Un hombre posando solo ante un monumento | Male speech, man speaking | Posible o Improbable, según lo que sugiera la escena |
| Un coche aparcado en una calle vacía | Car alarm | Improbable, fuente inactiva |
| Un coche aparcado en una calle vacía | Engine starting | Improbable, fuente inactiva |
| Una cocina sin nadie | Dishes, pots, and pans | Improbable, fuente inactiva |
| Una playa sin vehículos | Traffic noise, roadway noise | No, no hay fuente |
| Árboles en un día claramente ventoso | Rustling leaves | Sí |
| Árboles en una foto sin indicios de viento | Rustling leaves | Posible |

Son orientativos. Si tu juicio en un caso concreto es otro, vale el tuyo.

### Al terminar

Pulsa «Descargar resultados» y envía el fichero `.csv` a quien coordina el estudio. Durante el estudio principal, no comentes casos concretos con otros anotadores.
