"""백엔드 프로토콜(인터페이스) 정의. 앱은 오직 이 프로토콜에만 의존하며, 구체 구현체는 같은 패키지에 함께 둔다.

이렇게 추상 인터페이스로 분리해 두면, 실제 ML 모델 백엔드(jina-clip, vLLM 등)와
모델 없이 동작하는 mock 백엔드를 동일한 코드 경로로 교체해 끼울 수 있다.

세 가지 역할(role):
  * Embedder  — 이미지와 텍스트를 공유된 L2 정규화 벡터 공간으로 변환 (검색의 핵심).
  * Captioner — 대표 이미지를 짧은 한국어 설명으로 변환 (메타데이터이자 RAG 문서).
  * ChatLLM   — 한국어 질의를 정제하고, 검색 결과에 근거한 한국어 요약을 작성.
"""

# 'from __future__ import annotations'는 반드시 모듈의 첫 코드 라인이어야 한다.
# 타입 힌트를 문자열로 지연 평가하게 만들어, 순환 import나 미설치 의존성에 대한 부담을 줄인다.
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

# TYPE_CHECKING 블록 안의 import는 런타임에는 실행되지 않고 타입 검사 시에만 보인다.
# FolderHit를 런타임에 직접 import하면 core.models <-> backends 사이에 순환 import가 생길 수 있어 회피한다.
if TYPE_CHECKING:  # 런타임 import 순환을 피하기 위함
    from imgsearch.core.models import FolderHit


# @runtime_checkable: isinstance(obj, Embedder)로 덕 타이핑 검사를 가능하게 한다(메서드 시그니처는 검사하지 않음).
@runtime_checkable
class Embedder(Protocol):
    """이미지/텍스트를 동일한 임베딩 벡터 공간으로 매핑하는 임베더 프로토콜.

    검색의 핵심: 이미지와 질의 텍스트를 같은 공간에 두면 코사인 유사도로 교차 검색이 가능하다.
    구현체는 클래스 변수 ``name``(백엔드 식별자)과 ``dim``(벡터 차원)을 노출해야 한다.
    """

    name: str  # 백엔드 식별자 (예: "mock", "jina-clip")
    dim: int  # 출력 벡터의 차원 수

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        """이미지 경로들을 임베딩한다. 반환은 (n, dim) float32이며 L2 정규화되어 있고, 순서는 ``paths``와 일치한다."""

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """텍스트들을 임베딩한다. 반환은 (n, dim) float32이며 L2 정규화되어 있고, 순서는 ``texts``와 일치한다."""


@runtime_checkable
class Captioner(Protocol):
    """대표 이미지를 짧은 한국어 캡션으로 만드는 캡셔너 프로토콜."""

    name: str  # 백엔드 식별자

    def caption(self, image_path: str, context: str = "") -> str:
        """이미지에 대한 짧은 한국어 캡션을 반환한다. ``context``에는 사이드카(부가) 텍스트가 담길 수 있다."""


@runtime_checkable
class ChatLLM(Protocol):
    """한국어 질의 정제와 결과 요약을 담당하는 대화형 LLM 프로토콜."""

    name: str  # 백엔드 식별자

    def refine_query(self, ko_text: str) -> str:
        """한국어 질의를 정리/확장하여 임베딩에 사용할 검색 텍스트로 만든다."""

    def summarize(self, query: str, hits: "list[FolderHit]") -> str:
        """검색된 결과(hits)에 근거한 짧은 한국어 답변을 작성한다."""
