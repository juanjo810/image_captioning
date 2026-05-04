"""Minimal script to test Grounding DINO integration.

This does NOT yet integrate the full pipeline (scene, HOI, fusion). It only
validates that detections are produced correctly from a real image.
"""

from pathlib import Path

from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter


def main():
    base = Path("/home/jovyan/projects")

    adapter = GroundingDINOAdapter(
        config_path=base / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path=base / "models/groundingdino_swint_ogc.pth",
    )

    image_path = base / "data/test.jpg"

    prompt = "person, horse, cart, tree, building"

    detections = adapter.predict(image_path, prompt)

    print("\n=== DETECTIONS ===")
    for d in detections:
        print({
            "label": d.label,
            "bbox": [round(x, 1) for x in d.bbox],
            "confidence": round(d.confidence, 3),
        })


if __name__ == "__main__":
    main()
