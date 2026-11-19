# Estudio humano de plausibilidad

La evaluación automática compara las predicciones con pseudo-referencias derivadas de Visual Genome mediante las mismas reglas que usan los sistemas. Mide acuerdo con esas reglas, no si un sonido es plausible en la imagen. Este estudio aporta una referencia independiente: personas que juzgan, sin saber qué sistema propuso qué, si cada sonido se oiría en la escena, si cada objeto citado como evidencia se ve de verdad y si la etiqueta de escena predicha describe el lugar.

Este documento tiene dos partes: el protocolo para quien coordina el estudio y la guía para quien anota.

## Parte 1. Protocolo (coordinación)

### Qué se anota

Para cada imagen se construye un *pool* con:

- todos los nodos AudioSet predichos por cualquiera de los sistemas comparados;
- los nodos de la pseudo-referencia de esa imagen;
- dos sonidos al azar del vocabulario del estudio, que no proponía nadie para esa imagen. No son una trampa: dan la tasa de plausibilidad de un sonido cualquiera, que sirve de línea base;
- los términos visuales que algún sistema citó como evidencia, más un término al azar;
- las etiquetas de escena que predijo cada fuente de escena para esa imagen: los dos clasificadores Places365 de Workflow B y los dos VLM de Workflow A.

Cada anotador responde tres preguntas por imagen:

1. **Lugar:** ¿describe la etiqueta el tipo de lugar de la foto? (Sí / En parte / No).
2. **Objetos:** ¿se ve en la imagen? (Sí / No / Dudoso).
3. **Sonidos:** en una grabación de unos 10 segundos hecha en el lugar y el momento de la foto, ¿se oiría? (Sí / Posible / Improbable / No). Con «Improbable» o «No» se pide el motivo: no hay fuente, fuente inactiva, no encaja con el lugar, u otro.

La pregunta de lugar es la comprobación humana del paso de escena. La métrica automática de consistencia de escena depende de 13 grupos, 9 familias y unos pesos fijados a mano; el juicio humano sobre la etiqueta no depende de nada de eso y permite ver si la métrica automática lo sigue.

### Cómo se eligen las imágenes

- **Población:** las imágenes de la evaluación automática (3.000 de Visual Genome).
- **Selección:** un solo sorteo sobre toda la población, aleatorio, con semilla fija (`--seed`) y estratificado con asignación proporcional. No hay subconjuntos intermedios.
- **Estratos:** el grupo de escena de la imagen cruzado con que su pseudo-referencia esté vacía o no.
  - El grupo de escena es uno de los 13 de la métrica de escena, tal como lo asigna la etiqueta Places365 que predice `--strata-scene-system` (por defecto `owlv2/densenet161`, el clasificador DenseNet161). Garantiza que la muestra reproduce la mezcla de tipos de escena de la evaluación y permite desglosar los resultados por grupo.
  - La referencia vacía o no vacía cambia qué mide la métrica: con referencia vacía, abstención; con referencia no vacía, acierto.
- **Asignación:** a cada estrato le corresponden las imágenes proporcionales a su tamaño, redondeando por restos mayores. La muestra queda autoponderada: las medias de la muestra estiman las de la población sin pesos.
- **Piloto:** se saca igual, entre las imágenes que no entraron en el estudio.
- No se filtra por contenido, por dificultad ni por el resultado de ningún sistema. Tampoco se excluyen las imágenes en que algún sistema falló: en el análisis esa imagen no cuenta para ese sistema.

El estrato usa la predicción de un clasificador, no una verdad de escena, porque Visual Genome no la tiene. Para estratificar basta: una variable de estratificación no necesita ser correcta, solo conocerse antes del sorteo para todas las imágenes.

Con asignación proporcional los grupos poco frecuentes en Visual Genome quedan con pocas imágenes. `sampling_strata.csv` dice cuántas hay de cada estrato en la población, en el estudio y en el piloto; esa tabla va al apéndice del artículo. Las listas de identificadores están en `study_main.json` y `study_pilot.json` y conviene publicarlas: son lo que hace reproducible la muestra.

### Preparar el paquete

```bash
python -m scripts.evaluation.build_plausibility_pool \
  --manifest outputs/manifests/3000/predictions_manifest_3000.csv \
  --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \
  --scene-per-image outputs/final_metrics/stats_3000/scene_per_image.csv \
  --image-dir ../data/subsets/vg_5000_images \
  --n-images 300 --n-pilot 30 --seed 0 \
  --output-dir outputs/plausibility_v2
```

`--scene-per-image` es el CSV por imagen de `evaluate_scene_awareness.py`. De él salen el grupo de escena de los estratos y las etiquetas de escena que se juzgan. Sin esa opción el sorteo se estratifica solo por referencia vacía y no hay pregunta de lugar. `--image-ids` restringe las candidatas a una lista; por defecto son todas las de `--per-image`.

Genera en `--output-dir`:

| Fichero | Para quién | Qué es |
| --- | --- | --- |
| `images/` | anotadores | Copia de las imágenes seleccionadas. |
| `annotate_pilot.html` | anotadores | Página de anotación del piloto (30 imágenes distintas de las del estudio). |
| `annotate_main.html` | anotadores | Página de anotación del estudio (300 imágenes). |
| `study_*.json` | nadie en particular | Los ítems que lleva dentro cada página. No contienen nombres de sistema. |
| `key_*.json` | **solo coordinación** | Qué sistema predijo cada nodo y con qué evidencia; qué nodos son de referencia o aleatorios; qué fuente predijo cada etiqueta de escena; el estrato de cada imagen. Hace falta para el análisis. **No se entrega a los anotadores.** |
| `sampling_strata.csv` | coordinación | Imágenes por estrato en la población, en el estudio y en el piloto. |

Con los mismos argumentos el generador produce exactamente los mismos ficheros, así que se puede volver a ejecutar para actualizar las páginas sin que cambien los ítems ni las claves. Si la carpeta `images/` de destino tiene imágenes de otra selección, se detiene: cada diseño va en su carpeta.

### Comprobar que la muestra es representativa

Antes de repartir, se compara la muestra con la población en lo que la evaluación ya sabe de cada imagen:

```bash
python -m scripts.evaluation.sample_representativeness \
  --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \
  --scene-per-image outputs/final_metrics/stats_3000/scene_per_image.csv \
  --sample study_300=outputs/plausibility_v2/study_main.json \
  --sample pilot_30=outputs/plausibility_v2/study_pilot.json \
  --output-dir outputs/plausibility_v2/representativeness
```

Compara la proporción de referencias vacías, los nodos de referencia por imagen, las categorías AudioSet de primer nivel que toca la referencia, el F1 automático y el número de nodos de cada sistema, y el grupo de escena. Para cada característica da la media en la población, la media en la muestra y un p de Monte Carlo: la proporción de subconjuntos aleatorios del mismo tamaño cuya media se aleja de la población al menos tanto como la de la muestra. Con unas cuarenta características, dos o tres por debajo de 0,05 son lo esperable por azar; el resumen las cuenta. En las variables de los estratos la muestra coincide con la población por construcción; lo informativo es el resto. La tabla de `summary.md` sirve tal cual para el apéndice del artículo.

### Anotadores

En el estudio del artículo anotan dos de los autores. Es una limitación y hay que declararla. Lo que la acota:

- **No saben qué están juzgando.** Ninguno ve qué sistema propuso cada sonido, ni cuáles son de la pseudo-referencia o aleatorios, ni qué fuente predijo cada etiqueta de escena. Un sesgo a favor del trabajo propio no puede dirigirse a sabiendas a un sistema concreto.
- **Orden distinto.** Cada anotador ve las imágenes y los ítems barajados con su código.
- **Anotación independiente.** No comentan casos durante el estudio. Se reporta el acuerdo entre ellos (alfa de Krippendorff).
- **Línea base de azar.** Los sonidos aleatorios miden cuánto tiende cada anotador a decir que sí: si los juzgan plausibles a menudo, los juicios son laxos.
- **Regla conservadora.** Con dos anotadores, un sonido cuenta como plausible solo si lo dicen los dos, y lo mismo para objetos y etiquetas de escena. Se reporta también la lectura contraria, en la que basta uno. Cualquier forma de resolver los desacuerdos da un resultado entre esas dos, así que juntas muestran cuánto dependen las conclusiones de los ítems en que no coinciden.
- **Resultados ocultos hasta el final.** Mientras dura la anotación, los anotadores solo ven el acuerdo y la lista de desacuerdos del piloto. Las tablas por sistema no se les enseñan hasta que los dos han terminado el estudio.
- **Datos publicados.** Las respuestas de cada anotador se publican con el artículo.

Lo que no se puede descartar: que los dos compartan una idea de qué es plausible que otras personas no tendrían. Afecta a los valores absolutos más que a las comparaciones entre sistemas. Si una tercera persona puede juzgar después solo los ítems en desacuerdo, sin ver las respuestas, se añade sin repetir nada. Es el protocolo con que se etiquetó AudioSet: tres evaluadores por segmento y voto por mayoría, y el tercero solo se pedía cuando los dos primeros no coincidían (Gemmeke et al., 2017, sección 3.1).

Requisitos: leer inglés con soltura, porque los nombres y las descripciones de los sonidos y las etiquetas de lugar están en inglés. No hace falta formación en audio: se pregunta por plausibilidad cotidiana.

En el artículo se cuenta cuántos son, que son autores y qué parte del sistema conocía cada uno, su formación para la tarea (la guía y el piloto), el tiempo por imagen y el acuerdo.

### Repartir

A cada anotador se le entrega una carpeta con `annotate_pilot.html`, `annotate_main.html` y la carpeta `images/` al lado, tal como salen. Para sacarla del contenedor se empaqueta sin las claves:

```bash
cd outputs/plausibility_v2
python -m zipfile -c paquete_anotadores.zip annotate_pilot.html annotate_main.html images
```

El zip se descarga y se descomprime en el ordenador del anotador. La página se abre desde ahí con doble clic, en un navegador de escritorio (Chrome, Edge o Firefox); no necesita instalación ni conexión. La dirección del navegador tiene que empezar por `file:///`.

No funciona abierta desde Jupyter ni desde dentro del zip. Jupyter aísla las páginas que sirve, y el navegador bloquea entonces el guardado del progreso y las descargas; la página lo detecta y lo avisa. Dentro del zip no encuentra las imágenes, y también lo avisa.

Cada anotador recibe un código (por ejemplo `A1`, `A2`) y lo usa siempre. La página baraja el orden de las imágenes y de los ítems con ese código, de modo que cada anotador ve un orden distinto y reproducible.

### Fases

1. **Piloto.** Los anotadores hacen las 30 imágenes del piloto y envían su CSV. Se revisan juntos los desacuerdos y se ajusta esta guía si hace falta. Las respuestas del piloto no entran en los resultados.
2. **Estudio.** Cada anotador hace las 300 imágenes por su cuenta, sin comentar casos con los demás, y envía su CSV.
3. **Análisis.** Acuerdo entre anotadores, consenso y métricas por sistema y por fuente de escena.

### Qué devuelve cada anotador

El botón «Descargar resultados» genera `plausibility_v2_main_<código>.csv`, con una fila por respuesta:

| Columna | Contenido |
| --- | --- |
| `study_id`, `annotator` | Identificador del estudio y código del anotador. |
| `position` | Posición en que ese anotador vio la imagen. |
| `image_id` | Imagen. |
| `kind` | `scene`, `object` o `sound`. |
| `item_id`, `item_name` | Etiqueta de escena, término visual, o identificador y nombre del nodo AudioSet. |
| `answer` | Lugar: `yes`, `partly`, `no`. Objetos: `yes`, `no`, `unsure`. Sonidos: `1` Sí, `2` Posible, `3` Improbable, `4` No. |
| `reason` | Solo en sonidos con `3` o `4`: `absent`, `inactive`, `place`, `other`. |
| `image_seconds`, `image_done` | Tiempo acumulado en la imagen y si quedó completa. |

### Analizar

Los CSV que devuelven los anotadores se guardan en una carpeta (por ejemplo `outputs/plausibility_v2/results/`) y se analizan con la clave. Los ficheros se pasan por nombre, para que no entre ninguno de prueba:

```bash
python -m scripts.evaluation.analyze_plausibility \
  --annotations outputs/plausibility_v2/results/plausibility_v2_pilot_A1.csv \
                outputs/plausibility_v2/results/plausibility_v2_pilot_A2.csv \
  --key outputs/plausibility_v2/key_pilot.json \
  --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \
  --scene-per-image outputs/final_metrics/stats_3000/scene_per_image.csv \
  --output-dir outputs/plausibility_v2/analysis_pilot
```

Para el estudio se usan los ficheros `plausibility_v2_main_*` y `key_main.json`. El análisis se ejecuta dos veces: tal cual (`--ties negative`, la regla conservadora) y con `--ties positive` en otra carpeta de salida.

Cómo decide el análisis:

- Solo cuentan las imágenes que el anotador dejó completas.
- Una imagen entra en el consenso cuando la completaron al menos `--min-annotators` anotadores (2 por defecto; con `1` se pueden mirar los resultados de un solo anotador).
- El consenso de cada ítem es la mayoría estricta de quienes lo valoraron. Un empate se resuelve con `--ties`: `negative` (por defecto) lo cuenta como no plausible, no presente o no correcto; `positive`, como lo contrario. Con dos anotadores, un empate es un desacuerdo. El resumen dice cuántos hubo.
- «Plausible» se lee de dos maneras y se calculan las dos: amplia («Sí» o «Posible») y estricta (solo «Sí»). La amplia es la principal, porque la tarea es proponer sonidos plausibles, no sonidos seguros.
- Una etiqueta de escena es correcta en la lectura estricta con «Sí» y en la amplia también con «En parte».
- Si un sistema no tiene predicción para una imagen (ejecución fallida), esa imagen no cuenta para él. Las comparaciones entre dos sistemas usan las imágenes en que ambos tienen predicción.

Escribe en `--output-dir`:

| Fichero | Contenido |
| --- | --- |
| `summary.md` | Todas las tablas. |
| `annotators.csv` | Imágenes completadas, tiempo y distribución de respuestas de cada anotador. |
| `agreement.csv` | Alfa de Krippendorff con su intervalo: sonidos (escala de cuatro niveles y sus dos lecturas binarias), objetos y etiquetas de escena. Con tres o más anotadores, también por pares. |
| `items_consensus.csv` | Una fila por ítem, con la respuesta de cada anotador y el consenso. |
| `systems.csv` | Métricas por sistema con intervalos. Incluye `pseudo_reference` (la pseudo-referencia puntuada como un sistema más) y `random_baseline` (los sonidos al azar). |
| `comparisons.csv` | Diferencias pareadas entre sistemas. Por defecto, las mismas comparaciones que la evaluación automática; `--compare A:B` las sustituye. |
| `by_scene_group.csv` | Precisión, cobertura y F1 de cada sistema dentro de cada grupo de escena de los estratos, con el número de imágenes. |
| `scene_sources.csv` | Por fuente de escena: proporción de etiquetas juzgadas correctas, estricta y amplia, con intervalos. |
| `scene_comparisons.csv` | Diferencias pareadas entre fuentes de escena. |
| `scene_metric_vs_human.csv` | Solo con `--scene-per-image`: media de la métrica automática de escena cuando la etiqueta se juzgó correcta y cuando no, y AUC con intervalo. Un AUC de 0,5 significa que la métrica no distingue una etiqueta correcta de una incorrecta. |
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

Krippendorff propone 0,80 como umbral de fiabilidad y 0,667 como mínimo para sacar conclusiones tentativas. Con 30 imágenes el intervalo del alfa es ancho, así que el valor se lee junto a su intervalo. El alfa de objetos y el de etiquetas de escena se miran igual, pero no deciden: si salen bajos, se aclara su parte de la guía.

### Qué se reporta

Los resultados principales se fijan antes de anotar:

1. Precisión validada por humanos de cada sistema, con la lectura amplia.
2. Cobertura sobre el pool, que equivale a contar los sonidos plausibles que encuentra cada sistema por imagen.
3. Fidelidad de la evidencia.
4. Proporción de etiquetas de escena correctas por fuente.

El F1 contra el consenso humano se reporta como resumen, no como resultado principal. Premia a los sistemas que proponen más sonidos: los sonidos plausibles que solo propone uno entran en el pool y cuentan como fallos de cobertura de los demás. Tampoco es comparable en valor absoluto con el F1 contra pseudo-referencia, porque el conjunto de referencia es otro.

Lista completa:

- F1 contra el consenso humano: media por imagen del F1 entre los nodos predichos y los sonidos del pool juzgados plausibles, con la misma convención que la métrica automática (predicción y referencia vacías cuentan 1).
- Precisión validada por humanos de cada sistema: proporción de sus nodos juzgados plausibles, con la lectura amplia y con la estricta.
- Cobertura: proporción de los sonidos plausibles del pool que cada sistema recupera. El pool solo contiene lo que propuso algún sistema o la pseudo-referencia, así que es una cota superior de la cobertura real y hay que decirlo.
- Abstención: en las imágenes donde ningún sonido del pool se juzgó plausible, proporción en que el sistema no predice nada.
- Fidelidad de la evidencia: proporción de nodos cuyos términos citados se ven en la imagen.
- Origen de los errores: cruce entre evidencia vista y sonido plausible. Separa los errores de percepción (el objeto citado no está en la imagen) de las reglas que sobregeneran (el objeto está, pero el sonido no es plausible).
- Validez de la pseudo-referencia: proporción de sus nodos juzgados plausibles, y correlación por imagen entre el F1 contra pseudo-referencia y el F1 contra el juicio humano.
- Escena: proporción de etiquetas juzgadas correctas por fuente (Places365 y VLM), con diferencias pareadas, y si la métrica automática de escena separa las etiquetas correctas de las incorrectas.
- Desglose por grupo de escena, diciendo cuántas imágenes tiene cada grupo.
- Motivos de implausibilidad por sistema y por nodo (fuente ausente, inactiva, lugar).
- Línea base: tasa de plausibilidad de los sonidos aleatorios.
- Todo lo anterior con las dos reglas de empate.

Los intervalos de confianza se calculan remuestreando imágenes, no juicios, porque los ítems de una misma imagen no son independientes.

## Parte 2. Guía para anotadores

Gracias por participar. Vas a ver fotografías y, para cada una, unas etiquetas de lugar, una lista de objetos y una lista de sonidos. No hay respuestas correctas ni preguntas trampa: interesa tu juicio.

### Antes de empezar

- Usa un ordenador, no el móvil, con el navegador Chrome, Edge o Firefox.
- Descomprime el paquete y abre el fichero `annotate_pilot.html` (o `annotate_main.html`) con doble clic. La carpeta `images` tiene que estar junto al fichero.
- Escribe el código que te han dado y úsalo siempre.
- Puedes dejarlo y seguir otro día en el mismo navegador: el progreso se guarda solo. Al terminar cada sesión pulsa «Copia de seguridad» y conserva el fichero, por si acaso.
- Reserva sesiones de no más de una hora. Si te cansas, para.

### Tarea 1: lugar

Verás una o varias etiquetas de tipo de lugar, en inglés (por ejemplo «kitchen» o «street»). Para cada una, indica si describe el lugar de la foto.

- **Sí:** es ese tipo de lugar.
- **En parte:** se le parece, o solo describe una parte de la foto. Por ejemplo, «restaurant» para la cocina de un restaurante, o «forest path» para un bosque sin camino.
- **No:** es otro tipo de lugar.

Juzga cada etiqueta por separado: puede haber varias correctas, o ninguna. Si no conoces una palabra, búscala; si aun así no sabes qué lugar es, responde lo que mejor refleje tu duda.

### Tarea 2: objetos

Para cada objeto de la lista, indica si **se ve** en la imagen.

- **Sí:** se ve, aunque sea pequeño o esté parcialmente tapado.
- **No:** no aparece.
- **Dudoso:** no se distingue bien, o no sabes si lo que ves cuenta como ese objeto.

No importa si el objeto es relevante o no; solo si está.

### Tarea 3: sonidos

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
