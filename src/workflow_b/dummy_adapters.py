from src.geometry import Detection


class DummyDetector:
    def predict(self):
        return [
            Detection("person", [50, 50, 200, 300], 0.95),
            Detection("horse", [200, 100, 400, 350], 0.88),
        ]


class DummyScene:
    def predict(self):
        return "rural field", 0.7


class DummyHOI:
    def predict(self):
        return [
            {
                "human_bbox": [50, 50, 200, 300],
                "object_bbox": [200, 100, 400, 350],
                "verb": "ride",
                "confidence": 0.9
            }
        ]
