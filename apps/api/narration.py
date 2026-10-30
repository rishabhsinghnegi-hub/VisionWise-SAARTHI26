"""Optional OpenRouter narration for YOLO labels; never sends camera imagery."""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("visionwise.narration")


class Narrator:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = os.getenv("OPENROUTER_MODEL", "openrouter/auto").strip()
        self._cache: dict[tuple[tuple[str, str], ...], tuple[float, str]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def narrate(self, detections: list[dict]) -> str | None:
        if not self.configured or not detections:
            return None

        signature = tuple(sorted((item["label"], item["position"]) for item in detections))
        cached = self._cache.get(signature)
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]

        # Only detection metadata leaves the laptop. Image frames are never sent.
        scene = [
            {"object": item["label"], "confidence": round(item["confidence"], 2), "image_position": item["position"]}
            for item in detections[:30]
        ]
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 75,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Convert the provided object detections into one concise spoken scene notice. "
                        "Start with 'Notice:'. State only listed objects and their left, center, or right "
                        "image positions. Do not infer distance, movement, danger, collision, or safe paths. "
                        "Do not give instructions or navigation advice. If confidence is below 0.5, say "
                        "'possible'. Use at most 20 words."
                    ),
                },
                {"role": "user", "content": f"Detected objects: {scene}"},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=3.0)) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-Title": "VisionWise",
                    },
                    json=payload,
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                return None
            message = " ".join(content.split())[:240]
            if not message:
                return None
            self._cache[signature] = (time.monotonic(), message)
            self._expire_cache()
            return message
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            logger.warning("OpenRouter narration failed; using the local description")
            return None

    def _expire_cache(self) -> None:
        now = time.monotonic()
        self._cache = {key: value for key, value in self._cache.items() if now - value[0] < 60}
