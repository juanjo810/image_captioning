from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib import request, error
import json

from src.workflow_a.base import BaseVLM


class LlamaCppServerAdapter(BaseVLM):
    """OpenAI-compatible llama.cpp server adapter for multimodal Workflow A."""

    def __init__(
        self,
        model_id: str = "llamacpp-local-vlm",
        base_url: str = "http://localhost:8889",
        timeout: int = 600,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        image_path: str | Path | None,
        prompt: str,
        max_new_tokens: int = 1024,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path is not None:
            image_url = self._image_to_data_url(Path(image_path))
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": max_new_tokens,
        }

        response = self._post_json(
            endpoint="/v1/chat/completions",
            payload=payload,
        )

        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected llama.cpp response format: {response}"
            ) from exc

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP error {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not connect to llama.cpp server at {url}: {exc}") from exc

        return json.loads(body)

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/jpeg"

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
