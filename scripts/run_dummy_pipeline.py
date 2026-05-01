from src.workflow_b.dummy_adapters import DummyDetector, DummyScene, DummyHOI
from src.fusion import build_from_modules


def main():
    det = DummyDetector().predict()
    scene, conf = DummyScene().predict()
    hoi = DummyHOI().predict()

    core, extended = build_from_modules(
        image_id="test_image",
        width=640,
        height=480,
        detections=det,
        scene_label=scene,
        scene_conf=conf,
        hoi=hoi
    )

    print("CORE:\n", core.model_dump())
    print("\n---\n")
    print("EXTENDED:\n", extended.model_dump())


if __name__ == "__main__":
    main()
