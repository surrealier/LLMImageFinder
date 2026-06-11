"""Backend factory: turn the config's string choices into concrete objects.

Real backends are imported lazily inside ``try`` so the app always starts; if a requested
real backend is unavailable (missing deps, no model, unreachable endpoint) we fall back to
the deterministic mock and record a human-readable reason on ``config`` for the UI banner.
"""

from __future__ import annotations

from typing import Optional, Tuple

from imgsearch.backends.base import Captioner, ChatLLM, Embedder
from imgsearch.backends.mock import MockCaptioner, MockChatLLM, MockEmbedder
from imgsearch.config import AppConfig
from imgsearch.logging_setup import get_logger

log = get_logger("registry")


def build_embedder(cfg: AppConfig) -> Embedder:
    if cfg.embedder_backend == "jina-clip":
        try:
            from imgsearch.backends.jina_clip import JinaClipEmbedder

            emb = JinaClipEmbedder(
                model_id=cfg.embedder_model,
                device=cfg.embedder_device,
                dim=cfg.embed_dim,
            )
            log.info("Embedder: jina-clip (%s) dim=%d", cfg.embedder_model, emb.dim)
            return emb
        except Exception as e:
            cfg.mark_degraded(f"임베딩 모델 로드 실패({cfg.embedder_model}): {e}")
            log.warning("jina-clip unavailable, using mock embedder: %s", e)
    return MockEmbedder(dim=cfg.embed_dim)


def build_captioner(cfg: AppConfig) -> Optional[Captioner]:
    if not cfg.caption_enabled:
        return None
    if cfg.vlm_backend == "vllm":
        try:
            from imgsearch.backends.openai_compat import OpenAICompatVLMCaptioner

            cap = OpenAICompatVLMCaptioner(
                base_url=cfg.vlm_base_url, model=cfg.vlm_model, api_key=cfg.vlm_api_key
            )
            log.info("Captioner: vLLM (%s @ %s)", cfg.vlm_model, cfg.vlm_base_url)
            return cap
        except Exception as e:
            cfg.mark_degraded(f"VLM 캡션 백엔드 연결 실패: {e}")
            log.warning("vLLM captioner unavailable, using mock: %s", e)
    return MockCaptioner()


def build_chat(cfg: AppConfig) -> ChatLLM:
    if cfg.chat_backend == "vllm":
        try:
            from imgsearch.backends.openai_compat import OpenAICompatChatLLM

            chat = OpenAICompatChatLLM(
                base_url=cfg.chat_base_url, model=cfg.chat_model, api_key=cfg.chat_api_key
            )
            log.info("Chat: vLLM (%s @ %s)", cfg.chat_model, cfg.chat_base_url)
            return chat
        except Exception as e:
            cfg.mark_degraded(f"채팅 sLLM 백엔드 연결 실패: {e}")
            log.warning("vLLM chat unavailable, using mock: %s", e)
    return MockChatLLM()


def build_backends(
    cfg: AppConfig,
) -> Tuple[Embedder, Optional[Captioner], ChatLLM]:
    """Reset degraded state and build all three backends."""
    cfg.degraded = False
    cfg.degraded_reason = ""
    embedder = build_embedder(cfg)
    captioner = build_captioner(cfg)
    chat = build_chat(cfg)
    return embedder, captioner, chat
