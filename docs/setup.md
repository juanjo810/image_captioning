# SETUP

Manual oficial de instalacion para `image_captioning`.

Este documento describe una instalacion limpia en Ubuntu Linux. La guia asume que el proyecto se instalara bajo `/home/jovyan/projects/`, que es el layout que varios scripts del repositorio esperan actualmente.

> [!IMPORTANT]
> El repositorio esta dividido intencionalmente en tres entornos Conda independientes: Workflow A, Workflow B y Evaluation. No los fusiones. Cada entorno usa versiones distintas de Python, PyTorch y CUDA porque combina modelos y librerias con requisitos incompatibles.

## 1. Introduccion

`image_captioning` es un repositorio de investigacion para convertir imagenes en representaciones semanticas estructuradas y evaluar esas salidas con referencias visuales y pseudo-referencias acustico-semanticas basadas en la AudioSet ontology.

La arquitectura esta separada en tres bloques:

| Bloque | Proposito | Por que tiene entorno propio |
| --- | --- | --- |
| Workflow A | Generar JSON estructurado con un VLM servido por llama.cpp. | Es solo un cliente HTTP: se mantiene aparte para no arrastrar el stack pesado de Workflow B ni competir con `llama-server` por la GPU. |
| Workflow B | Ejecutar Places365, GroundingDINO/OWLv2, geometria, fusion y HOI opcional. | GroundingDINO, UPT y Pocket son sensibles a versiones antiguas de PyTorch/CUDA. |
| Evaluation | Calcular metricas de caption, Visual Genome, semantic metrics y AudioSet. | Algunas metricas usan dependencias propias como `pycocoevalcap`, Java/SPICE y librerias de evaluacion. |

Filosofia del repositorio:

- mantener pipelines reproducibles aunque requieran entornos separados
- documentar dependencias externas en vez de esconderlas
- no versionar datasets, checkpoints ni outputs grandes
- distinguir claramente lo implementado, lo parcial y lo legacy

## 2. Requisitos del Sistema

Sistema recomendado:

| Requisito | Recomendacion |
| --- | --- |
| OS | Ubuntu Linux 22.04 o similar |
| Git | Version reciente |
| Conda | Miniconda o Anaconda |
| GPU | NVIDIA GPU con memoria suficiente para los modelos |
| Driver NVIDIA | Compatible con CUDA 11.8 y CUDA 13.0 |
| RAM | 32 GB recomendado; 16 GB minimo para pruebas pequeñas |
| Disco | 80-150 GB libres si se descargan checkpoints, datasets y caches HF |
| Internet | Necesario para clonar repos y descargar modelos |

Comprobar GPU y driver:

```bash
nvidia-smi
```

Comprobar Conda:

```bash
conda --version
```

Si Conda no esta instalado, instala Miniconda desde la documentacion oficial y reinicia la shell.

## 3. Layout Recomendado

Varios scripts actuales usan rutas bajo `/home/jovyan/projects`. Usa este layout para evitar cambios manuales:

```text
/home/jovyan/projects/
  image_captioning/
  GroundingDINO/
  upt/
  pocket/
  llama.cpp/
  models/
    groundingdino_swint_ogc.pth
    places365/
      categories_places365.txt
      resnet50_places365.pth.tar
      densenet161_places365.pth.tar
    upt/
      upt-r50-hicodet.pt
    gguf/
      <vlm-model>.gguf
      <mmproj-model>.gguf
  datasets/
    visual_genome/
    hico_20160224_det/
```

> [!NOTE]
> El script `scripts/run_grounding_dino_pipeline.py` define `base = Path("/home/jovyan/projects")`. Por eso GroundingDINO, Places365 y UPT deben estar ahi salvo que modifiques el script para tu maquina.

Crear la carpeta base:

```bash
mkdir -p /home/jovyan/projects
cd /home/jovyan/projects
```

## 4. Clonar el Repositorio

```bash
cd /home/jovyan/projects
git clone <repo-url> image_captioning
cd image_captioning
```

Comprobar estructura minima:

```bash
ls README.md scripts src metrics docs ontology.json
```

## 5. Resumen de Entornos

Estas combinaciones han sido verificadas experimentalmente en este repositorio:

| Entorno | Proposito | Python | Torch | CUDA | GPU requerida | Verificado |
| --- | --- | --- | --- | --- | --- | --- |
| `imagecap-a` | Workflow A, cliente llama.cpp (Gemma/Qwen servidos via `llama-server`) | 3.11 | no requerido | — | No (la usa `llama-server`) | Si |
| `imagecap-b` | Workflow B, Places365, GroundingDINO, OWLv2, UPT | 3.10 | 2.5.1+cu118 | 11.8 | Si | Si |
| `imagecap-eval` | Caption metrics, VG metrics, AudioSet metrics, SPICE | 3.10 | 2.12.0+cu130 | 13.0 | Recomendada | Si |

No mezcles los entornos. La razón principal es que Workflow B depende de proyectos externos sensibles a ABI/versiones, mientras que Workflow A y Evaluation se benefician de stacks mas recientes.

### 5.1 PYTHONNOUSERSITE

Aplica a los tres entornos. Antes de trabajar en cualquiera de ellos:

```bash
export PYTHONNOUSERSITE=1
```

Motivo: si el sistema tiene paquetes instalados en el user site (`~/.local/lib/python3.X/site-packages`),
Python los antepone a los del entorno Conda activo. Un `torch` o un `transformers` de ahi puede
ensombrecer al del entorno y romper imports de forma dificil de diagnosticar, porque `conda list`
sigue mostrando la version correcta. El riesgo es mayor en `imagecap-b` e `imagecap-eval`, que son
los entornos con el stack pesado.

`PYTHONNOUSERSITE=1` obliga a Python a ignorar ese directorio y hace la ejecucion reproducible.
Para dejarlo fijado en un entorno concreto:

```bash
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate <entorno>
```

Comprobar que esta surtiendo efecto:

```bash
python -c "import site; print(site.ENABLE_USER_SITE)"   # debe imprimir False
```

## 6. Instalacion de Workflow B

Workflow B es el entorno mas delicado. Instala primero este entorno si tu objetivo es ejecutar detectores visuales.

### 6.1 Crear el Entorno

```bash
conda create -n imagecap-b python=3.10 -y
conda activate imagecap-b
python -m pip install --upgrade pip setuptools wheel
```

Instalar PyTorch verificado para CUDA 11.8:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Verificar:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

Salida esperada:

```text
torch: 2.5.1+cu118
cuda: 11.8
cuda available: True
```

### 6.2 Dependencias Python del Proyecto

Instala dependencias comunes:

```bash
cd /home/jovyan/projects/image_captioning
pip install pillow numpy scipy pandas scikit-learn opencv-python pydantic python-dotenv tqdm matplotlib requests
pip install transformers==4.41.2 huggingface_hub safetensors timm supervision yacs pycocotools
```

Alternativa equivalente y mas reproducible: `requirements_workflowb.txt` esta generado desde el
entorno `imagecap-b` verificado y cubre esos mismos paquetes con versiones exactas, incluido torch
(su primera linea anade el indice `cu118`, asi que tambien sustituye al `pip install torch...` de
6.1):

```bash
pip install -r requirements_workflowb.txt
```

No incluye `groundingdino`, `pocket` ni `upt`: los tres se instalan en editable desde sus repos
clonados (secciones 6.3, 6.4 y 6.5), y congelar sus rutas locales en el fichero no funcionaria en
otra maquina.

### 6.3 GroundingDINO

Clonar:

```bash
cd /home/jovyan/projects
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

Verificar import:

```bash
python - <<'PY'
import groundingdino
print("GroundingDINO import ok")
PY
```

Checkpoint esperado por el script:

```text
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
```

Crear carpeta:

```bash
mkdir -p /home/jovyan/projects/models
```

Descarga el checkpoint `groundingdino_swint_ogc.pth` desde la release oficial de GroundingDINO y guardalo exactamente en esa ruta.

### 6.4 Pocket

UPT depende de Pocket. Instalarlo en editable:

```bash
cd /home/jovyan/projects
git clone https://github.com/fredzzhang/pocket.git
cd pocket
pip install -e .
```

Verificar:

```bash
python - <<'PY'
import pocket
print("pocket import ok")
PY
```

### 6.5 UPT

Clonar UPT:

```bash
cd /home/jovyan/projects
git clone https://github.com/fredzzhang/upt.git
cd upt
pip install -e .
```

Rutas esperadas por `scripts/run_grounding_dino_pipeline.py`:

```text
/home/jovyan/projects/upt
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
/home/jovyan/projects/upt/hicodet
```

Crear carpeta de checkpoints:

```bash
mkdir -p /home/jovyan/projects/models/upt
```

Descarga `upt-r50-hicodet.pt` y guardalo en:

```text
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
```

> [!TIP]
> UPT solo hace falta para la ruta legacy con HOI (`--legacy-visual-core --hoi upt`). El default del script es `--hoi none`, y en la ruta por defecto (audioset) el flag no tiene ningun efecto, asi que puedes instalar y probar Workflow B sin UPT.

### 6.6 Places365

Crear carpetas:

```bash
mkdir -p /home/jovyan/projects/models/places365
mkdir -p /home/jovyan/projects/places365
```

Archivos esperados:

```text
/home/jovyan/projects/places365/categories_places365.txt
/home/jovyan/projects/models/places365/resnet50_places365.pth.tar
/home/jovyan/projects/models/places365/densenet161_places365.pth.tar
```

Descarga `categories_places365.txt` y los checkpoints de Places365 desde las fuentes oficiales del proyecto Places365.

### 6.7 OWLv2

OWLv2 se carga desde Hugging Face:

```text
google/owlv2-base-patch16-ensemble
```

Comprobar carga basica:

```bash
python - <<'PY'
from transformers import Owlv2ForObjectDetection, Owlv2Processor
model_id = "google/owlv2-base-patch16-ensemble"
processor = Owlv2Processor.from_pretrained(model_id)
model = Owlv2ForObjectDetection.from_pretrained(model_id)
print("OWLv2 load ok")
PY
```

### 6.8 Smoke Tests de Workflow B

Desde la raiz del repo:

```bash
cd /home/jovyan/projects/image_captioning
python -m py_compile scripts/run_grounding_dino_pipeline.py
```

Comprobar imports principales:

```bash
python - <<'PY'
from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter
from src.workflow_b.owlv2_adapter import OWLv2Adapter
from src.workflow_b.places365_adapter import Places365Adapter
print("workflow_b imports ok")
PY
```

Ejecutar una imagen con OWLv2 y sin HOI:

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector owlv2 \
  --output-dir outputs/workflow_b/owlv2_resnet50 \
  --verbose
```

Ejecutar con GroundingDINO:

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --output-dir outputs/workflow_b/grounding_dino_resnet50 \
  --verbose
```

Nota: invoca los scripts **como módulo** (`python -m scripts.<nombre>`). Llamarlos por ruta
(`python -m scripts.run_grounding_dino_pipeline`) falla con `ModuleNotFoundError: No module
named 'src...'`, porque Python pone `scripts/` en el path en vez de la raíz del repo.

`--hoi` ya no hace falta aquí: su default es `none`, y solo tiene efecto junto con
`--legacy-visual-core`. Sin ese flag, la salida es el formato `AudioSetCoreJSON`.

Salida esperada:

```text
[OK] JSON saved to: outputs/workflow_b/<condition>/example.json
```

## 7. Instalacion de Workflow A

Workflow A usa VLMs para generar JSON estructurado. Su unico backend es un cliente de servidor llama.cpp OpenAI-compatible.

### 7.1 Crear el Entorno

```bash
conda create -n imagecap-a python=3.11 -y
conda activate imagecap-a
python -m pip install --upgrade pip setuptools wheel
```

Instalar dependencias del repo:

```bash
cd /home/jovyan/projects/image_captioning
pip install -r requirements_workflow_a.txt
```

Este entorno **no necesita PyTorch ni Transformers**. Al eliminarse el `Gemma4Adapter`, el unico
backend (`LlamaCppServerAdapter`) habla con `llama-server` por HTTP con la stdlib, asi que
`requirements_workflow_a.txt` instala solo `pydantic` y sus dependencias directas. El trabajo
pesado de GPU lo hace `llama-server`, que es un proceso aparte (ver 7.2) y no depende de este
entorno Conda.

### 7.2 llama.cpp
Para el montaje del servidor de llama.cpp y la configuración de modelos se ha seguido la documentación de Unsloth. Recomendamos seguir la documentación de su web: https://unsloth.ai/docs/models/qwen3.6#llama.cpp-guide 
No obstante, exponemos debajo los comandos más importantes para su instalación.
Clonar:

```bash
cd /home/jovyan/projects
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

Compilar con CUDA:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Verificar binario:

```bash
./build/bin/llama-server --help | head
```

### 7.3 Modelos GGUF y mmproj

Para VLMs en llama.cpp necesitas:

- un modelo `.gguf`
- un proyector multimodal `mmproj-*.gguf`

Layout recomendado:

```text
/home/jovyan/projects/models/gguf/
  qwen-vl-model.gguf
  mmproj-qwen-vl.gguf
  gemma-vl-model.gguf
  mmproj-gemma-vl.gguf
```

El repositorio no incluye estos archivos. Descargalos desde la fuente correspondiente al modelo que vayas a servir.

### 7.4 Gemma4 y Qwen

Workflow A tiene **un solo backend**: `LlamaCppServerAdapter`, cliente de un servidor llama.cpp
OpenAI-compatible. El antiguo `Gemma4Adapter` (backend Hugging Face directo) fue eliminado, y con
él los flags `--model` y `--temperature`.

Uso: sirve el modelo que quieras (Gemma4, Qwen…) en formato GGUF con `llama-server` y apunta
`--server-url` a él. `--model-id` solo nombra el modelo en la petición y en el `metadata` del JSON.

Importante: el adaptador envía únicamente `model`, `messages` y `max_tokens`. **Todo el sampling
—temperatura, top-p, top-k— se configura al lanzar `llama-server`** (ver 7.5), no desde el CLI del
script. Si el JSON sale errático, es ahí donde hay que ajustar.

### 7.5 Lanzar llama-server

Ejemplo para Gemma4:

```bash
./llama.cpp/llama-server \
    --model ~/projects/models/gemma4/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    --mmproj ~/projects/models/gemma4/mmproj-BF16.gguf \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --alias "unsloth/gemma-4-26B-A4B-it-GGUF" \
    --port 8889 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

Ejemplo para Qwen3.6:

```bash
./llama.cpp/llama-server \
    --model ~/projects/models/qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --mmproj ~/projects/models/qwen36/mmproj-F16.gguf \
    --temp 0.7 \
    --top-p 0.8 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --top-k 20 \
    --alias "unsloth/Qwen3.6-35B-A3B-GGUF" \
    --port 8889 \
    --chat-template-kwargs '{"enable_thinking":false}'
```



Comprobar que responde:

```bash
curl http://localhost:8889/v1/models
```

### 7.6 Smoke Tests de Workflow A

En otra terminal:

```bash
conda activate imagecap-a
export PYTHONNOUSERSITE=1
cd /home/jovyan/projects/image_captioning
python -m py_compile scripts/run_workflow_a.py
```

Ejecutar una imagen (formato `AudioSetCoreJSON`, el default):

```bash
python -m scripts.run_workflow_a \
  --image data/images/example.jpg \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

En la ruta legacy, con los nodos AudioSet del bloque `acoustic_semantics`:

```bash
python -m scripts.run_workflow_a \
  --image data/images/example.jpg \
  --server-url http://localhost:8889 \
  --legacy-visual-core \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

`--include-audioset-nodes` **solo tiene efecto junto con `--legacy-visual-core`**: en la ruta por
defecto los nodos AudioSet se piden y validan siempre.

Salida esperada:

```text
outputs/workflow_a/json/example.json
outputs/workflow_a/captions/example.txt
outputs/workflow_a/raw/example.txt
```

Con `--legacy-visual-core`, todo eso cuelga de `outputs/workflow_a/legacy/`.

## 8. Entorno de Evaluation

Este entorno calcula metricas de captions, Visual Genome, semantic metrics y AudioSet.

### 8.1 Crear el Entorno

```bash
conda create -n imagecap-eval python=3.10 -y
conda activate imagecap-eval
python -m pip install --upgrade pip setuptools wheel
```

Instalar dependencias:

```bash
cd /home/jovyan/projects/image_captioning
pip install -r requirements_evaluation.txt
```

`requirements_evaluation.txt` esta generado desde el entorno verificado e incluye torch: su primera
linea anade el indice de PyTorch para CUDA 13.0, asi que las builds `+cu130` se resuelven solas y no
hace falta instalar torch aparte. Si prefieres hacerlo en dos pasos:

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

### 8.2 pycocoevalcap y Caption Metrics

`requirements_evaluation.txt` incluye `pycocoevalcap`. Se usa para metricas de caption como BLEU, METEOR, ROUGE, CIDEr y SPICE cuando el script correspondiente lo requiere.

Verificar import:

```bash
python - <<'PY'
import pycocoevalcap
print("pycocoevalcap import ok")
PY
```

### 8.3 SPICE y Java 8

SPICE requiere Java 8. Java >=17 suele producir errores por cambios de compatibilidad en librerias antiguas usadas por SPICE.

Comprobar Java:

```bash
java -version
```

Instalar Java 8 en Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jre
```

Seleccionar Java 8 si tienes varias versiones:

```bash
sudo update-alternatives --config java
java -version
```

Workaround si no puedes cambiar Java globalmente: define `JAVA_HOME` para Java 8 en la terminal donde ejecutes metricas SPICE.

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

### 8.4 Visual Genome y Semantic Evaluation

La evaluacion semantica usa:

- manifest CSV de predicciones
- referencias de objetos de Visual Genome
- referencias de relaciones de Visual Genome
- alias maps
- `ontology.json` para AudioSet

Validar scripts:

```bash
python -m py_compile scripts/evaluation/evaluate_all_semantic_metrics.py
python -m scripts.evaluation.validate_audioset_ontology
```

Salida esperada de AudioSet:

```text
total_nodes: 632
root_nodes: 7
missing_child_refs: 0
cycle_nodes: 0
```

## 9. Descarga de Modelos

### 9.1 GroundingDINO

Guardar:

```text
/home/jovyan/projects/models/groundingdino_swint_ogc.pth
```

Tambien debe existir:

```text
/home/jovyan/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

### 9.2 OWLv2

Modelo Hugging Face:

```text
google/owlv2-base-patch16-ensemble
```

Se descarga en la cache de Hugging Face al primer uso.

### 9.3 Places365

Guardar:

```text
/home/jovyan/projects/places365/categories_places365.txt
/home/jovyan/projects/models/places365/resnet50_places365.pth.tar
/home/jovyan/projects/models/places365/densenet161_places365.pth.tar
```

### 9.4 UPT

Guardar:

```text
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
```

Dataset esperado por UPT:

```text
/home/jovyan/projects/upt/hicodet
```

### 9.5 Gemma4 y Qwen

Gemma y Qwen se usan actualmente a traves de llama.cpp si esta convertido a GGUF con su `mmproj` correspondiente.

Guardar GGUFs en:

```text
/home/jovyan/projects/models/gguf/
```

## 10. Datasets

### 10.1 Visual Genome

Layout recomendado:

```text
/home/jovyan/projects/datasets/visual_genome/
  region_descriptions.json
  objects.json
  relationships.json
  images/
```

El repositorio no incluye Visual Genome.

**La API REST de `visualgenome.org/api/v0/...` está muerta** (HTTP 404, deprecada desde 2023). La
vía que funciona son los dumps estáticos:

```text
https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/
```

Descarga `image_data.json` (ids + urls) y después `objects.json`, `relationships.json` y
`region_descriptions.json`. Los ficheros `object_alias.txt` y `relationship_alias.txt` de esa misma
URL ya vienen en el formato `canonical,alias1,...` que espera `vg_utils.load_alias_map`.

#### Subconjunto de trabajo

Los dumps completos son enormes y las pruebas se pueden hacer sobre un subconjunto (por ejemplo, 100
imágenes). Al recortarlos, filtra `objects.json`, `relationships.json` y `region_descriptions.json`
a los ids elegidos y **normaliza `name` → `names`** en el sujeto y el objeto de cada relación:

```python
for node in (rel.get("subject", {}), rel.get("object", {})):
    if "names" not in node and node.get("name"):
        node["names"] = [node["name"]]
```

`build_vg_references.py` lee solo la clave `names`, mientras que el dump de VG guarda `name`
(singular) en la gran mayoría de los sujetos. Sin esa normalización,
`vg_relationship_references.csv` sale casi vacío y las métricas de relaciones dan ~0.

Comprobación rápida tras generar las referencias: cuenta cuántas filas traen la columna
`relationships` vacía. Que algunas lo estén es normal —hay imágenes de VG sin relaciones
anotadas—; que lo estén casi todas es el síntoma de este problema.

### 10.2 Manifests

La evaluacion principal espera CSV con columnas minimas:

```csv
image_id,json_path,detector,scene_model,captioner
```

Construir manifest desde JSONs:

```bash
python -m scripts.evaluation.build_prediction_manifest \
  --inputs outputs/workflow_b/owlv2_resnet50 outputs/workflow_b/grounding_dino_resnet50 \
  --output outputs/manifests/predictions.csv
```

### 10.3 AudioSet Ontology

El archivo versionado es:

```text
ontology.json
```

Validarlo:

```bash
python -m scripts.evaluation.validate_audioset_ontology
```

### 10.4 Alias Maps

Rutas recomendadas:

```text
data/aliases/object_alias.csv
data/aliases/relationship_alias.csv
```

### 10.5 Referencias Generadas

Construir referencias Visual Genome:

```bash
python -m scripts.evaluation.build_vg_references \
  --manifest outputs/manifests/predictions.csv \
  --region-descriptions /home/jovyan/projects/datasets/visual_genome/region_descriptions.json \
  --objects /home/jovyan/projects/datasets/visual_genome/objects.json \
  --relationships /home/jovyan/projects/datasets/visual_genome/relationships.json \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --output-dir data/vg_refs
```

Genera:

```text
data/vg_refs/vg_caption_references.csv
data/vg_refs/vg_object_references.csv
data/vg_refs/vg_relationship_references.csv
```

## 11. Ejecutar el Proyecto

### 11.1 Workflow A

```bash
conda activate imagecap-a
export PYTHONNOUSERSITE=1
cd /home/jovyan/projects/image_captioning

python -m scripts.run_workflow_a \
  --image data/images/example.jpg \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Directorio de imagenes `.jpg`:

```bash
python -m scripts.run_workflow_a \
  --image-dir data/images \
  --limit 20 \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Para lotes largos sin supervision, `run_workflow_a_with_retries` acepta los mismos argumentos y
reintenta solo las imagenes que fallaron la validacion:

```bash
python -m scripts.run_workflow_a_with_retries \
  --image-dir data/images \
  --server-url http://localhost:8889 \
  --max-retries 2 \
  --output-dir outputs/workflow_a
```

La ruta de generacion (audioset por defecto, o legacy con `--legacy-visual-core`) y los modos
`--call-mode single|three|two` se describen en [workflow_a.md](workflow_a.md).

### 11.2 Workflow B

```bash
conda activate imagecap-b
cd /home/jovyan/projects/image_captioning

python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector owlv2 \
  --output-dir outputs/workflow_b/owlv2_resnet50
```

Con GroundingDINO:

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

Lote sobre un directorio (los modelos se cargan una sola vez):

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image-dir data/images \
  --limit 100 \
  --detector grounding_dino \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

La ruta legacy (`--legacy-visual-core`, con `--hoi upt` y relaciones espaciales) se describe en
[workflow_b.md](workflow_b.md).

### 11.3 Evaluation

Metricas oficiales AudioSet, en dos pasos — scorer y recorte a las columnas oficiales:

```bash
conda activate imagecap-eval
cd /home/jovyan/projects/image_captioning

python -m scripts.evaluation.evaluate_audioset_semantics \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --output outputs/metrics/audioset_summary.csv

python -m scripts.evaluation.filter_metrics_csv \
  --input outputs/metrics/audioset_summary.csv \
  --output outputs/metrics/audioset_official.csv \
  --metric-group audioset
```

Metricas de caption (CIDEr/SPICE, CLIPScore, CHAIR):

```bash
python -m scripts.evaluation.evaluate_all_caption_metrics \
  --manifest outputs/manifests/predictions.csv \
  --caption-refs data/vg_refs/vg_caption_references.csv \
  --image-dir data/images \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --output outputs/metrics/caption_summary.csv
```

El orquestador legacy solo vale para predicciones `CoreJSON` (`--legacy-visual-core`); apuntado al
formato audioset devuelve ceros sin avisar:

```bash
python -m scripts.evaluation.evaluate_all_semantic_metrics \
  --manifest outputs/manifests/predictions_legacy.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --label-space workflow_b \
  --audioset-pred-source core_rules \
  --output outputs/metrics/semantic_summary.csv \
  --per-image-output outputs/metrics/semantic_per_image.csv
```

Detalle completo de cada script en [evaluation.md](evaluation.md).

## 12. Smoke Tests

### 12.1 CUDA

Ejecutar en `imagecap-b` e `imagecap-eval` (`imagecap-a` no instala Torch):

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### 12.2 Imports del Repositorio

```bash
cd /home/jovyan/projects/image_captioning
python - <<'PY'
from metrics.audioset_ontology import load_audioset_ontology
ontology = load_audioset_ontology()
print("AudioSet nodes:", len(ontology.node_by_id))
PY
```

### 12.3 GroundingDINO

En `imagecap-b`:

```bash
python - <<'PY'
from groundingdino.util.inference import load_model
print("GroundingDINO utilities import ok")
PY
```

### 12.4 Workflow A

En `imagecap-a`, con llama-server activo:

```bash
curl http://localhost:8889/v1/models
python -m py_compile scripts/run_workflow_a.py
```

### 12.5 Workflow B

En `imagecap-b`:

```bash
python -m py_compile scripts/run_grounding_dino_pipeline.py
python -m scripts.run_dummy_pipeline
```

`run_dummy_pipeline` no carga ningun modelo: ejercita el contrato de fusion con fixtures, asi que
sirve para comprobar el repo aunque falten checkpoints.

### 12.6 Evaluation

En `imagecap-eval`:

```bash
python -m py_compile scripts/evaluation/evaluate_all_semantic_metrics.py
python -m scripts.evaluation.validate_audioset_ontology
```

## 13. Troubleshooting

### CUDA Mismatch

Sintomas:

- `torch.cuda.is_available()` devuelve `False`
- errores `CUDA driver version is insufficient`
- errores al cargar extensiones C++/CUDA

Soluciones:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Instala la build de torch indicada para el entorno activo. No reutilices el torch de otro entorno.

### Torch Mismatch

Si `pip install -r ...` cambia torch, reinstala la version verificada del entorno. Ejemplo para Workflow B:

```bash
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### GroundingDINO Import Errors

Comprueba:

```bash
conda activate imagecap-b
python -c "import groundingdino; print('ok')"
```

Si falla, reinstala desde `/home/jovyan/projects/GroundingDINO`:

```bash
cd /home/jovyan/projects/GroundingDINO
pip install -e .
```

### Pocket Installation

Si UPT falla con imports de `pocket`:

```bash
cd /home/jovyan/projects/pocket
pip install -e .
python -c "import pocket; print('ok')"
```

### UPT Installation

Si falla `--hoi upt`, primero prueba sin HOI:

```bash
python -m scripts.run_grounding_dino_pipeline \
  --image data/images/example.jpg \
  --detector owlv2 \
  --hoi none \
  --output-dir outputs/workflow_b/debug
```

Despues revisa:

```text
/home/jovyan/projects/upt
/home/jovyan/projects/models/upt/upt-r50-hicodet.pt
/home/jovyan/projects/upt/hicodet
```

### llama.cpp

Si `curl http://localhost:8889/v1/models` falla:

- confirma que `llama-server` sigue corriendo
- revisa el puerto
- revisa que el modelo y `--mmproj` existan
- comprueba memoria GPU disponible con `nvidia-smi`

### PYTHONNOUSERSITE

Si aparecen imports desde `~/.local`, activa:

```bash
export PYTHONNOUSERSITE=1
python -c "import site; print(site.ENABLE_USER_SITE)"
```

Salida esperada:

```text
False
```

### Java y SPICE

Si SPICE falla con Java moderno, usa Java 8:

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

### Hugging Face Authentication

Si Gemma4, OWLv2 o Qwen requieren acceso:

```bash
huggingface-cli login
```

Tambien puedes definir una cache local:

```bash
export HF_HOME=/home/jovyan/projects/models/huggingface_cache
```

### OOM

Sintomas:

- `CUDA out of memory`
- proceso terminado por el sistema

Soluciones:

- usa un modelo GGUF mas pequeno o mas cuantizado
- reduce batch/numero de imagenes
- cierra otros procesos GPU
- usa `--hoi none` para descartar UPT
- prueba OWLv2 antes que GroundingDINO+UPT

### Missing Checkpoints

Revisa rutas exactas:

```bash
ls /home/jovyan/projects/models/groundingdino_swint_ogc.pth
ls /home/jovyan/projects/models/places365/resnet50_places365.pth.tar
ls /home/jovyan/projects/places365/categories_places365.txt
```

### Visual Genome Paths

Errores comunes:

- `image_id` del manifest no coincide con refs
- `json_path` relativo se ejecuta desde otro directorio
- CSV de relaciones no usa `subject::predicate::object`

Solucion: ejecuta evaluacion desde la raiz del repo y revisa los CSV con `head`.

```bash
head outputs/manifests/predictions.csv
head data/vg_refs/vg_object_references.csv
head data/vg_refs/vg_relationship_references.csv
```

### Permisos Linux

Si no puedes escribir en `/home/jovyan/projects`:

```bash
whoami
ls -ld /home/jovyan/projects
```

Cambia ownership solo si eres administrador del sistema:

```bash
sudo chown -R "$USER":"$USER" /home/jovyan/projects
```

## 14. FAQ

### Puedo usar un solo entorno?

No es recomendable. Workflow B depende de combinaciones especificas de PyTorch/CUDA y repos externos. Mezclarlo con Workflow A o Evaluation suele romper dependencias.

### Puedo instalar en otra carpeta que no sea `/home/jovyan/projects`?

Si, pero algunos scripts tienen rutas esperadas bajo `/home/jovyan/projects`. Para una primera instalacion reproducible, usa el layout recomendado.

### OWLv2 necesita checkpoint local?

No necesariamente. Se descarga desde Hugging Face y queda en cache.

### Qwen esta implementado como backend directo?

No. El unico backend de Workflow A es el cliente de llama.cpp (`LlamaCppServerAdapter`): sirve el modelo en GGUF con `llama-server` y apunta `--server-url` a el. Los flags `--model` y `--temperature` ya no existen.

### Workflow B detecta audio?

No. Detecta evidencia visual. Las proyecciones AudioSet son semanticas y visualmente inferidas.

### Visual Genome es ground truth acustico?

No. Visual Genome se usa para referencias visuales y pseudo-referencias acustico-semanticas.

### Por que Java 8?

SPICE depende de tooling antiguo. Java >=17 suele romper esa ruta.

## 15. Checklist Final

Usa esta lista para confirmar que la instalacion esta completa:

- [ ] `nvidia-smi` detecta la GPU.
- [ ] Conda tiene los entornos `imagecap-a`, `imagecap-b` e `imagecap-eval`.
- [ ] `imagecap-a` usa Python 3.11.
- [ ] `imagecap-b` usa Python 3.10 y Torch 2.5.1+cu118.
- [ ] `imagecap-eval` usa Python 3.10 y Torch 2.12.0+cu130.
- [ ] `PYTHONNOUSERSITE=1` esta exportado en los entornos que uses (ver 5.1).
- [ ] GroundingDINO importa correctamente.
- [ ] Pocket importa correctamente si vas a usar UPT.
- [ ] Los checkpoints de GroundingDINO, Places365 y UPT estan en las rutas esperadas.
- [ ] `llama-server` responde en `http://localhost:8889/v1/models`.
- [ ] Workflow A genera JSON para una imagen.
- [ ] Workflow B genera JSON para una imagen (ruta audioset por defecto).
- [ ] `validate_audioset_ontology.py` reporta `missing_child_refs: 0` y `cycle_nodes: 0`.
- [ ] Java 8 esta disponible si vas a ejecutar SPICE.
- [ ] El manifest CSV y las referencias Visual Genome existen antes de evaluar.
- [ ] `evaluate_audioset_semantics.py` genera `audioset_summary.csv`, y `filter_metrics_csv.py --metric-group audioset` lo recorta a las 5 metricas oficiales.
