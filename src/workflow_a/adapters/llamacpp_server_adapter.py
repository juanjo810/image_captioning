from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Any
from urllib import request, error
import json

from src.workflow_a.base import BaseVLM


class LlamaCppServerAdapter(BaseVLM):
    """OpenAI-compatible llama.cpp server adapter for multimodal Workflow A.

    Sampling parameters (``temperature``, ``top_p``, ``seed``) are sent with
    every request when given, so that the values that produced a prediction
    are the ones recorded in its metadata rather than whatever the server
    happened to be started with. When they are left as ``None`` the server
    defaults apply; in that case ``server_info()`` (a best-effort GET of the
    server's ``/props`` endpoint, cached) reports those defaults together with
    the server build, so the effective configuration is still recorded.
    """

    def __init__(
        self,
        model_id: str = "llamacpp-local-vlm",
        base_url: str = "http://localhost:8889",
        timeout: int = 600,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._server_info: dict[str, Any] | None = None

    @property
    def sampling_params(self) -> dict[str, Any]:
        """The sampling fields actually sent with each request (None = not sent)."""
        params: dict[str, Any] = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.seed is not None:
            params["seed"] = self.seed
        return params

    def server_info(self) -> dict[str, Any]:
        """Best-effort snapshot of the llama.cpp server configuration from
        ``GET /props``: build identifier, loaded model, a SHA-256 fingerprint
        of the chat template (not its text) and the server-side default
        generation settings (temperature, top_p, seed, ...). Never
        raises: on any failure it returns ``{"error": ...}`` so a metadata
        block can still be written. Fetched once per adapter instance."""
        if self._server_info is not None:
            return self._server_info

        url = f"{self.base_url}/props"
        try:
            with request.urlopen(request.Request(url, method="GET"), timeout=30) as resp:
                props = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            self._server_info = {"error": f"{type(exc).__name__}: {exc}", "url": url}
            return self._server_info

        info: dict[str, Any] = {"url": url}
        for key in ("build_info", "model_path", "model_alias", "model", "modalities", "total_slots"):
            if key in props:
                info[key] = props[key]
        # The chat template is several kB of Jinja and identical for every
        # image of a run, so only a fingerprint goes into the per-image
        # metadata: enough to tell whether two runs used the same template.
        chat_template = props.get("chat_template")
        if isinstance(chat_template, str):
            info["chat_template_sha256"] = hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest()
            info["chat_template_chars"] = len(chat_template)
        defaults = props.get("default_generation_settings")
        if isinstance(defaults, dict):
            # Newer servers nest the sampling settings under "params"; older
            # ones put them at the top level. Keep whichever is there.
            info["default_generation_settings"] = defaults.get("params", defaults)
        info["props_keys"] = sorted(props.keys()) if isinstance(props, dict) else []
        self._server_info = info
        return info

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
            **self.sampling_params,
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
