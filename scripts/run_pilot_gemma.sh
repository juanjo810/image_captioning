#!/bin/bash

set -e

IN_BASE="outputs/pilot_vg/visual_json_spatialRelations"
OUT_BASE="outputs/pilot_vg/captions_gemma_spatialRelations"

CONFIGS=(
  "grounding_dino_resnet50"
  "grounding_dino_densenet161"
  "owlv2_resnet50"
  "owlv2_densenet161"
)

for CONFIG in "${CONFIGS[@]}"; do
  IN_DIR="${IN_BASE}/${CONFIG}"
  OUT_DIR="${OUT_BASE}/${CONFIG}"

  echo "========================================"
  echo "Generating Gemma captions for ${CONFIG}"
  echo "Input: ${IN_DIR}"
  echo "Output: ${OUT_DIR}"
  echo "========================================"

  python -m scripts.run_json_captioning \
    --input "$IN_DIR" \
    --output-dir "$OUT_DIR" \
    --model-id google/gemma-4-E4B-it
done
