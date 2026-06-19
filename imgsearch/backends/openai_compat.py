"""OpenAI 호환 백엔드(주 대상: 로컬 vLLM 서버).

vLLM은 OpenAI 호환 API를 제공하므로, 여기서는 httpx로 평범한 HTTP만 주고받는다.
같은 클라이언트가 Ollama나 임의의 OpenAI 호환 엔드포인트에도 그대로 동작한다. 생성자에서
``/models``를 핑(ping)하므로, 닿지 않는 서버라면 레지스트리가 mock으로 깔끔히 폴백한다.

Windows에서는 vLLM이 보통 WSL2/Docker/다른 호스트에서 돌아간다. 앱은 base URL만
알면 된다(예: ``http://localhost:8000/v1``).
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
    """이미지를 JPEG base64 문자열로 인코딩한다(VLM에 data URL로 인라인 전달하기 위함)."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as im:
        # EXIF 회전 정보를 실제 픽셀에 반영(exif_transpose) -> 모델이 똑바로 선 이미지를 보게 한다.
        im = ImageOps.exif_transpose(im.convert("RGB"))
        if max(im.size) > max_side:
            # 너무 큰 이미지는 긴 변 기준으로 축소(토큰/전송량 절감, 비율 유지).
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)  # 품질 88: 화질과 용량의 절충
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _OpenAICompatBase:
    """OpenAI 호환 엔드포인트와 통신하는 공통 베이스(HTTP 클라이언트 + chat 호출 래퍼)."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 120.0) -> None:
        """HTTP 클라이언트를 만들고, 생성 시점에 ``/models``를 호출해 서버 도달 가능성을 검증한다."""
        self.model = model
        base = base_url.rstrip("/") + "/"  # 끝 슬래시를 정규화해 상대 경로 결합이 일관되게 동작하도록
        self._client = httpx.Client(
            base_url=base,
            timeout=timeout,
            # vLLM은 키를 검사하지 않지만 OpenAI 호환 형식상 Authorization 헤더가 필요하다(빈 값이면 "EMPTY").
            headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        )
        # 연결성 프로브: 죽은 엔드포인트를 빌드 시점에 드러내기 위함. 짧은 타임아웃을 쓰는 이유는
        # 방화벽에 막히거나 닿지 않는 호스트가 앱 시작을 약 120초간 멈추게 두지 않으려는 것.
        self._client.get("models", timeout=5.0).raise_for_status()

    def _chat(self, messages: list, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """chat/completions를 호출하고 첫 번째 응답 메시지의 본문(앞뒤 공백 제거)을 반환한다."""
        resp = self._client.post(
            "chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()  # 4xx/5xx면 예외 -> 호출부의 try/except가 폴백을 담당
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class OpenAICompatVLMCaptioner(_OpenAICompatBase):
    """OpenAI 호환 VLM(예: Qwen2.5-VL)으로 이미지 캡션을 생성하는 캡셔너."""

    name = "vllm"

    def caption(self, image_path: str, context: str = "") -> str:
        """이미지를 base64로 인라인 첨부하고, 시스템/유저 프롬프트와 함께 VLM에 보내 한국어 캡션을 받는다."""
        b64 = _encode_image(image_path)
        user_text = prompts.CAPTION_USER
        if context:
            # 사이드카 메모가 있으면 프롬프트에 덧붙이되, 너무 길어지지 않도록 300자로 제한.
            user_text += f"\nReference note: {context[:300]}"
        messages = [
            {"role": "system", "content": prompts.CAPTION_SYSTEM},
            {
                "role": "user",
                # OpenAI 비전 형식: content를 텍스트 파트와 이미지 파트의 리스트로 구성한다.
                "content": [
                    {"type": "text", "text": user_text},
                    # 이미지는 data URL(base64 인라인)로 전달 -> 외부 호스팅 없이 바로 전송.
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]
        return self._chat(messages, max_tokens=160, temperature=0.2)


class OpenAICompatChatLLM(_OpenAICompatBase):
    """OpenAI 호환 텍스트 LLM으로 질의 정제·계획·요약을 수행하는 ChatLLM."""

    name = "vllm"

    def refine_query(self, ko_text: str) -> str:
        """한국어 질의를 LLM으로 정제한다. 실패하거나 빈 응답이면 원본 질의를 그대로 사용(검색이 멈추지 않도록)."""
        messages = [
            {"role": "system", "content": prompts.REFINE_SYSTEM},
            {"role": "user", "content": prompts.REFINE_USER.format(query=ko_text)},
        ]
        try:
            # temperature=0.0: 정제는 창의성이 아니라 일관성이 중요하므로 결정적으로 생성.
            out = self._chat(messages, max_tokens=80, temperature=0.0)
            return out or ko_text  # 빈 문자열이면 원본 질의로 폴백
        except Exception as e:
            log.warning("refine_query failed, using raw query: %s", e)
            return ko_text

    def plan(self, query: str, class_names=None) -> dict:
        """질의를 {semantic, required_objects, excluded_objects} JSON으로 분해한다.

        LLM 출력은 신뢰할 수 없으므로 코드펜스를 제거하고 try/except로 json.loads를 감싼다.
        호출부는 None/잘못된 반환을 '규칙 기반 계획으로 폴백'으로 처리하므로 여기서 절대 크래시하지 않는다.
        """
        import json

        # class_names는 dict(id->이름) 또는 리스트로 올 수 있어 둘 다 이름 리스트로 정규화한다.
        names = (
            list(class_names.values()) if isinstance(class_names, dict) else list(class_names or [])
        )
        messages = [
            {"role": "system", "content": prompts.PLAN_SYSTEM},
            {"role": "user", "content": prompts.PLAN_USER.format(
                # 클래스 목록이 비면 LLM이 빈 칸을 임의로 채우지 않도록 "(없음)"을 명시.
                classes=", ".join(str(n) for n in names) or "(없음)", query=query)},
        ]
        try:
            out = self._chat(messages, max_tokens=200, temperature=0.0)
        except Exception as e:
            log.warning("plan failed: %s", e)
            return {}  # 빈 dict -> 호출부가 규칙 기반 계획으로 폴백
        text = out.strip()
        if text.startswith("```"):  # ```json … ``` 코드펜스가 있으면 제거
            text = text.strip("`")
            # 펜스/언어 라벨을 떼어낸 뒤, 첫 '{'부터 마지막 '}'까지만 잘라 순수 JSON 본문을 추출.
            text = text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(text)
            # 파싱은 됐지만 dict가 아니면(예: 배열/숫자) 계획으로 쓸 수 없으므로 빈 dict로 취급.
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}  # 파싱 실패도 폴백 신호

    def summarize(self, query: str, hits: list[FolderHit]) -> str:
        """검색된 폴더 설명을 근거(context)로 LLM에 요약을 요청한다. 실패 시 상위 폴더명 나열로 폴백."""
        if not hits:
            return f"No images matched '{query}'."
        # 상위 8개 폴더의 (폴더명: 캡션)을 근거로 제공. 캡션이 없으면 "(설명 없음)"으로 표기.
        ctx = "\n".join(
            f"- {Path(h.folder).name}: {h.caption or '(설명 없음)'}" for h in hits[:8]
        )
        messages = [
            {"role": "system", "content": prompts.SUMMARY_SYSTEM},
            {"role": "user", "content": prompts.SUMMARY_USER.format(query=query, n=len(hits), context=ctx)},
        ]
        try:
            # temperature=0.3: 요약은 약간의 자연스러움을 허용하되 환각을 억제하기 위해 낮게 유지.
            return self._chat(messages, max_tokens=220, temperature=0.3)
        except Exception as e:
            log.warning("summarize failed: %s", e)
            # LLM이 죽어도 사용자에게는 최소한의 결과(상위 3개 폴더명)를 보여준다.
            tops = ", ".join(Path(h.folder).name for h in hits[:3])
            return f"Top {len(hits)} folders for '{query}': {tops}."
