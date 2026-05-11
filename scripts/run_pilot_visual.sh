#!/bin/bash

set -e

IMAGE_DIR="/home/jovyan/projects/data/vg_subset"
OUT_BASE="outputs/pilot_vg/visual_json"

CONFIGS=(
  #"grounding_dino resnet50"
  #"grounding_dino densenet161"
  "owlv2 resnet50"
  "owlv2 densenet161"
)

for CONFIG in "${CONFIGS[@]}"; do
  DETECTOR=$(echo $CONFIG | cut -d' ' -f1)
  SCENE_MODEL=$(echo $CONFIG | cut -d' ' -f2)

  OUT_DIR="${OUT_BASE}/${DETECTOR}_${SCENE_MODEL}"

  echo "========================================"
  echo "Running detector=${DETECTOR}, scene=${SCENE_MODEL}"
  echo "Output: ${OUT_DIR}"
  echo "========================================"

  mkdir -p "$OUT_DIR"

  for IMG in "$IMAGE_DIR"/*.{jpg,jpeg,png}; do
    [ -e "$IMG" ] || continue

    python -m scripts.run_grounding_dino_pipeline \
      --image "$IMG" \
      --hoi upt \
      --detector "$DETECTOR" \
      --scene-architecture "$SCENE_MODEL" \
      --output-dir "$OUT_DIR"
  done
done
