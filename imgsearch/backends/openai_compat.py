"""OpenAI-compatible backends (primary target: a local vLLM server).

vLLM serves an OpenAI-compatible API, so these talk plain HTTP via httpx. The same
client works against Ollama or any OpenAI-compatible endpoint. The constructor pings
``/models`` so an unreachable server makes the registry fall back to mock cleanly.

vLLM typically runs under WSL2/Docker/another host on Windows; the app only needs the
base URL (e.g. ``http://localhost:8000/v1``).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx

from imgsearch.backends import prompts
from imgsearch.core.models import FolderHit
from imgsearch.logging_setup import get_logger

log = get_logger("openai_compat")


def _encode_image(image_path: str, max_side: int = 1024) -> str:
    from PIL import Image, ImageOps

    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im.convert("RGB"))
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _OpenAICompatBase:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 120.0) -> None:
        self.model = model
        base = base_url.rstrip("/") + "/"
        self._client = httpx.Client(
            base_url=base,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        )
        # connectivity probe so a dead endpoint surfaces at build time — short
        # timeout so a filtered/unreachable host can't hang app startup ~120s.
        self._client.get("models", timeout=5.0).raise_for_status()

    def _chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.2) -> str:
        resp = self._client.post(
            "chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class OpenAICompatVLMCaptioner(_OpenAICompatBase):
    name = "vllm"

    def caption(self, image_path: str, context: str = "") -> str:
        b64 = _encode_image(image_path)
        user_text = prompts.CAPTION_USER
        if context:
            user_text += f"\n참고 메모: {context[:300]}"
        messages = [
            {"role": "system", "content": prompts.CAPTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]
        return self._chat(messages, max_tokens=160, temperature=0.2)


class OpenAICompatChatLLM(_OpenAICompatBase):
    name = "vllm"

    def refine_query(self, ko_text: str) -> str:
        messages = [
            {"role": "system", "content": prompts.REFINE_SYSTEM},
            {"role": "user", "content": prompts.REFINE_USER.format(query=ko_text)},
        ]
        try:
            out = self._chat(messages, max_tokens=80, temperature=0.0)
            return out or ko_text
        except Exception as e:
            log.warning("refine_query failed, using raw query: %s", e)
            return ko_text

    def plan(self, query: str, class_names=None) -> dict:
        """Decompose a query into {semantic, required_objects, excluded_objects} JSON.
        Untrusted output: strip fences, json.loads in try/except — the caller treats a
        None/invalid return as 'fall back to the rule-based plan', so never crashes."""
        import json

        names = (
            list(class_names.values()) if isinstance(class_names, dict) else list(class_names or [])
        )
        messages = [
            {"role": "system", "content": prompts.PLAN_SYSTEM},
            {"role": "user", "content": prompts.PLAN_USER.format(
                classes=", ".join(str(n) for n in names) or "(없음)", query=query)},
        ]
        try:
            out = self._chat(messages, max_tokens=200, temperature=0.0)
        except Exception as e:
            log.warning("plan failed: %s", e)
            return {}
        text = out.strip()
        if text.startswith("```"):  # strip a ```json … ``` fence if present
            text = text.strip("`")
            text = text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def summarize(self, query: str, hits: list[FolderHit]) -> str:
        if not hits:
            return f"'{query}'에 해당하는 이미지를 찾지 못했습니다."
        ctx = "\n".join(
            f"- {Path(h.folder).name}: {h.caption or '(설명 없음)'}" for h in hits[:8]
        )
        messages = [
            {"role": "system", "content": prompts.SUMMARY_SYSTEM},
            {"role": "user", "content": prompts.SUMMARY_USER.format(query=query, n=len(hits), context=ctx)},
        ]
        try:
            return self._chat(messages, max_tokens=220, temperature=0.3)
        except Exception as e:
            log.warning("summarize failed: %s", e)
            tops = ", ".join(Path(h.folder).name for h in hits[:3])
            return f"'{query}' 관련 상위 {len(hits)}개 폴더: {tops}."
