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
| Workflow A | Generar JSON estructurado con un VLM, normalmente servido por llama.cpp o cargado con Hugging Face. | Los VLM modernos y `transformers` recientes requieren versiones nuevas de Python/PyTorch. |
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
| `imagecap-a` | Workflow A, VLM, llama.cpp client, Gemma/Qwen via server | 3.11 | 2.11.0+cu130 | 13.0 | Recomendada | Si |
| `imagecap-b` | Workflow B, Places365, GroundingDINO, OWLv2, UPT | 3.10 | 2.5.1+cu118 | 11.8 | Si | Si |
| `imagecap-eval` | Caption metrics, VG metrics, AudioSet metrics, SPICE | 3.10 | 2.12.0+cu130 | 13.0 | Recomendada | Si |

No mezcles los entornos. La razón principal es que Workflow B depende de proyectos externos sensibles a ABI/versiones, mientras que Workflow A y Evaluation se benefician de stacks mas recientes.

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

`requirements_workflowb.txt` existe como snapshot historico, pero es muy grande e incluye dependencias editables. Para una instalacion limpia se recomienda instalar los bloques explicitos de esta guia.

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
> Para instalar y probar Workflow B sin UPT, usa `--hoi none`. El default actual del script es `--hoi upt`, por lo que conviene indicar `--hoi none` explicitamente en smoke tests.

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
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector owlv2 \
  --hoi none \
  --output-dir outputs/workflow_b/owlv2_resnet50 \
  --verbose
```

Ejecutar con GroundingDINO:

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --hoi none \
  --output-dir outputs/workflow_b/grounding_dino_resnet50 \
  --verbose
```

Salida esperada:

```text
[OK] JSON saved to: outputs/workflow_b/<condition>/example.json
```

## 7. Instalacion de Workflow A

Workflow A usa VLMs para generar JSON estructurado. El backend por defecto del script es `llamacpp`, apuntando a un servidor OpenAI-compatible.

### 7.1 Crear el Entorno

```bash
conda create -n imagecap-a python=3.11 -y
conda activate imagecap-a
python -m pip install --upgrade pip setuptools wheel
```

Instalar PyTorch verificado para CUDA 13.0:

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

Instalar dependencias del repo:

```bash
cd /home/jovyan/projects/image_captioning
pip install -r requirements_workflow_a.txt
```

Si el resolver intenta cambiar la version de torch, reinstala la version verificada:

```bash
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

### 7.2 PYTHONNOUSERSITE

Usa:

```bash
export PYTHONNOUSERSITE=1
```

Motivo: algunos entornos tienen paquetes instalados en el user site (`~/.local/lib/python...`) que se mezclan con Conda y rompen imports de `torch`, `transformers`, `pydantic` o librerias CUDA. `PYTHONNOUSERSITE=1` fuerza a Python a ignorar esos paquetes externos y hace la ejecucion mas reproducible.

Puedes dejarlo permanente para este entorno:

```bash
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate imagecap-a
```

### 7.3 llama.cpp
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

### 7.4 Modelos GGUF y mmproj

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

### 7.5 Gemma4 y Qwen

El script implementa `--model gemma4` como backend Hugging Face directo y `--model llamacpp` como cliente para un servidor local.

Uso recomendado:

- **Gemma4/Qwen via GGUF**: servir el modelo con llama.cpp y usar `--model llamacpp`

### 7.6 Lanzar llama-server

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

### 7.7 Smoke Tests de Workflow A

En otra terminal:

```bash
conda activate imagecap-a
export PYTHONNOUSERSITE=1
cd /home/jovyan/projects/image_captioning
python -m py_compile scripts/run_workflow_a.py
```

Ejecutar una imagen:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Con nodos AudioSet opcionales:

```bash
python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

Salida esperada:

```text
outputs/workflow_a/json/example.json
outputs/workflow_a/captions/example.txt
outputs/workflow_a/raw/example.txt
```

## 8. Entorno de Evaluation

Este entorno calcula metricas de captions, Visual Genome, semantic metrics y AudioSet.

### 8.1 Crear el Entorno

```bash
conda create -n imagecap-eval python=3.10 -y
conda activate imagecap-eval
python -m pip install --upgrade pip setuptools wheel
```

Instalar PyTorch verificado para CUDA 13.0:

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

Instalar dependencias:

```bash
cd /home/jovyan/projects/image_captioning
pip install -r requirements_evaluation.txt
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
python scripts/evaluation/validate_audioset_ontology.py
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

### 10.2 Manifests

La evaluacion principal espera CSV con columnas minimas:

```csv
image_id,json_path,detector,scene_model,captioner
```

Construir manifest desde JSONs:

```bash
python scripts/evaluation/build_prediction_manifest.py \
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
python scripts/evaluation/validate_audioset_ontology.py
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
python scripts/evaluation/build_vg_references.py \
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

python scripts/run_workflow_a.py \
  --image data/images/example.jpg \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --output-dir outputs/workflow_a
```

Directorio de imagenes `.jpg`:

```bash
python scripts/run_workflow_a.py \
  --image-dir data/images \
  --limit 20 \
  --model llamacpp \
  --server-url http://localhost:8889 \
  --include-audioset-nodes \
  --output-dir outputs/workflow_a
```

### 11.2 Workflow B

```bash
conda activate imagecap-b
cd /home/jovyan/projects/image_captioning

python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector owlv2 \
  --hoi none \
  --output-dir outputs/workflow_b/owlv2_resnet50
```

Con GroundingDINO:

```bash
python scripts/run_grounding_dino_pipeline.py \
  --image data/images/example.jpg \
  --detector grounding_dino \
  --hoi none \
  --output-dir outputs/workflow_b/grounding_dino_resnet50
```

### 11.3 Evaluation

```bash
conda activate imagecap-eval
cd /home/jovyan/projects/image_captioning

python scripts/evaluation/evaluate_all_semantic_metrics.py \
  --manifest outputs/manifests/predictions.csv \
  --vg-object-refs data/vg_refs/vg_object_references.csv \
  --vg-relationship-refs data/vg_refs/vg_relationship_references.csv \
  --object-alias data/aliases/object_alias.csv \
  --relationship-alias data/aliases/relationship_alias.csv \
  --label-space workflow_b \
  --audioset-pred-source core_rules \
  --output outputs/metrics/semantic_summary.csv \
  --per-image-output outputs/metrics/semantic_per_image.csv
```

## 12. Smoke Tests

### 12.1 CUDA

Ejecutar en cada entorno:

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
```

### 12.6 Evaluation

En `imagecap-eval`:

```bash
python -m py_compile scripts/evaluation/evaluate_all_semantic_metrics.py
python scripts/evaluation/validate_audioset_ontology.py
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
python scripts/run_grounding_dino_pipeline.py \
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

No. Actualmente se usa a traves de llama.cpp con `--model llamacpp`.

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
- [ ] `imagecap-a` usa Python 3.11 y Torch 2.11.0+cu130.
- [ ] `imagecap-b` usa Python 3.10 y Torch 2.5.1+cu118.
- [ ] `imagecap-eval` usa Python 3.10 y Torch 2.12.0+cu130.
- [ ] `PYTHONNOUSERSITE=1` esta activo en Workflow A.
- [ ] GroundingDINO importa correctamente.
- [ ] Pocket importa correctamente si vas a usar UPT.
- [ ] Los checkpoints de GroundingDINO, Places365 y UPT estan en las rutas esperadas.
- [ ] `llama-server` responde en `http://localhost:8889/v1/models`.
- [ ] Workflow A genera JSON para una imagen.
- [ ] Workflow B genera JSON para una imagen con `--hoi none`.
- [ ] `validate_audioset_ontology.py` reporta `missing_child_refs: 0` y `cycle_nodes: 0`.
- [ ] Java 8 esta disponible si vas a ejecutar SPICE.
- [ ] El manifest CSV y las referencias Visual Genome existen antes de evaluar.
- [ ] `evaluate_all_semantic_metrics.py` genera `semantic_summary.csv`.
