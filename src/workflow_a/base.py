from pathlib import Path


class BaseVLM:
    def generate(self, image_path: str | Path | None, prompt: str) -> str:
        """image_path=None sends a text-only prompt (no image attached)."""
        raise NotImplementedError
