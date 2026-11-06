#!/bin/bash

set -e

# Uso: scripts/run_pilot_workflow_b.sh [audioset|legacy]
#   audioset (por defecto): AudioSetCoreJSON, sin --legacy-visual-core (--hoi se ignora).
#   legacy: CoreJSON clásico, con --legacy-visual-core --hoi upt (comportamiento original).
MODE="${1:-audioset}"

IMAGE_DIR="${VG_SUBSET:-/home/jovyan/projects/data/subsets/vg_test_100}/images"

CONFIGS=(
  "grounding_dino resnet50"
  "grounding_dino densenet161"
  "owlv2 resnet50"
  "owlv2 densenet161"
)

case "$MODE" in
  audioset)
    OUT_BASE="outputs/pilot_vg/visual_json_audioset"
    EXTRA_ARGS=()
    ;;
  legacy)
    OUT_BASE="outputs/pilot_vg/visual_json"
    EXTRA_ARGS=(--legacy-visual-core --hoi upt)
    ;;
  *)
    echo "Modo desconocido: '$MODE' (usa 'audioset' o 'legacy')" >&2
    exit 1
    ;;
esac

for CONFIG in "${CONFIGS[@]}"; do
  DETECTOR=$(echo $CONFIG | cut -d' ' -f1)
  SCENE_MODEL=$(echo $CONFIG | cut -d' ' -f2)

  OUT_DIR="${OUT_BASE}/${DETECTOR}_${SCENE_MODEL}"

  echo "========================================"
  echo "Modo: ${MODE} | detector=${DETECTOR}, scene=${SCENE_MODEL}"
  echo "Output: ${OUT_DIR}"
  echo "========================================"

  mkdir -p "$OUT_DIR"

  python -m scripts.run_grounding_dino_pipeline \
    --image-dir "$IMAGE_DIR" \
    --detector "$DETECTOR" \
    --scene-architecture "$SCENE_MODEL" \
    --output-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
done
