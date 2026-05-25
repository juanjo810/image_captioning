from pathlib import Path


class BaseVLM:
    def generate(self, image_path: str | Path, prompt: str) -> str:
        raise NotImplementedError
