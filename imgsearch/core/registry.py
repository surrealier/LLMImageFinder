"""백엔드 팩토리: 설정의 '문자열 선택값'을 실제 객체로 변환한다.

실제(real) 백엔드는 ``try`` 블록 안에서 '지연 import' 한다. 이렇게 해야 무거운 의존성
(예: 모델 라이브러리)이 깔려 있지 않아도 앱 자체는 항상 기동된다. 요청한 실제 백엔드를
쓸 수 없으면(의존성 없음 · 모델 없음 · 엔드포인트 접속 불가 등) 결정론적(deterministic)
mock으로 폴백하고, 사람이 읽을 수 있는 사유를 ``config``에 기록해 UI 상단 배너로 안내한다.

즉, '항상 동작하는 mock-first' 설계 — 환경이 완비되면 실제 백엔드로, 아니면 mock으로
자동 강등(degrade)된다.
"""

from __future__ import annotations

from typing import Optional, Tuple

from imgsearch.backends.base import Captioner, ChatLLM, Embedder
from imgsearch.backends.mock import MockCaptioner, MockChatLLM, MockEmbedder
from imgsearch.config import AppConfig
from imgsearch.logging_setup import get_logger

log = get_logger("registry")


def build_embedder(cfg: AppConfig) -> Embedder:
    """설정에 따라 임베더를 만든다. 실패하면 MockEmbedder로 폴백.

    jina-clip을 요청했고 로드에 성공하면 그것을, 그렇지 않으면(요청 안 함 / 로드 실패)
    동일 차원의 mock 임베더를 돌려준다. 따라서 반환값이 None이 되는 일은 없다.
    """
    if cfg.embedder_backend == "jina-clip":
        try:
            # 무거운 모델 의존성은 이 시점에만 import — 미설치 환경에서도 앱 기동을 보장.
            from imgsearch.backends.jina_clip import JinaClipEmbedder

            emb = JinaClipEmbedder(
                model_id=cfg.embedder_model,
                device=cfg.embedder_device,
                dim=cfg.embed_dim,
            )
            log.info("Embedder: jina-clip (%s) dim=%d", cfg.embedder_model, emb.dim)
            return emb
        except Exception as e:
            # 로드 실패 사유를 config에 남겨 UI 배너로 사용자에게 보여주고, mock으로 강등.
            cfg.mark_degraded(f"임베딩 모델 로드 실패({cfg.embedder_model}): {e}")
            log.warning("jina-clip unavailable, using mock embedder: %s", e)
    # 폴백: mock도 동일한 embed_dim을 써야 스토어 차원과 어긋나지 않는다.
    return MockEmbedder(dim=cfg.embed_dim)


def build_captioner(cfg: AppConfig) -> Optional[Captioner]:
    """이미지 캡션 생성기(VLM)를 만든다. 캡션 기능이 꺼져 있으면 None을 반환.

    임베더/채팅과 달리 캡션은 '선택' 기능이므로(끄면 색인 시 캡션을 안 만든다) None을
    돌려줄 수 있다. 켜져 있는데 실제 백엔드 연결에 실패하면 MockCaptioner로 폴백한다.
    """
    if not cfg.caption_enabled:
        return None  # 기능 자체가 비활성 -> 캡셔너 없음
    if cfg.vlm_backend == "vllm":
        try:
            # vLLM은 OpenAI 호환 API로 노출되므로 OpenAI 호환 클라이언트로 접속한다.
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
    """질의 정제·요약·계획에 쓰는 채팅 LLM을 만든다. 실패하면 MockChatLLM로 폴백.

    채팅은 항상 객체가 필요하므로(없으면 정제/요약을 건너뛰는 로직이 None 체크에 의존)
    절대 None을 반환하지 않고 최소한 mock을 돌려준다.
    """
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
    """강등(degraded) 상태를 초기화하고 세 백엔드를 한꺼번에 만든다.

    앱 시작/설정 변경 시 호출되는 진입점. 먼저 이전 강등 표시를 리셋해야, 이번 빌드에서
    실제로 발생한 폴백만 UI 배너에 반영된다(과거 사유가 남지 않게).
    """
    # 직전 빌드의 강등 흔적을 깨끗이 지우고 새로 판정한다.
    cfg.degraded = False
    cfg.degraded_reason = ""
    embedder = build_embedder(cfg)
    captioner = build_captioner(cfg)
    chat = build_chat(cfg)
    return embedder, captioner, chat
