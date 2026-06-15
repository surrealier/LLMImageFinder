"""GraphStore 프로토콜 + 그래프를 구축(build)할 때 넣는 입력 레코드 정의.

이 프로토콜은 서로 교체 가능한 두 백엔드가 구현한다: 순수 파이썬 인메모리 저장소
(항상 사용 가능한 기본값)와 임베디드 kuzu 저장소(진짜 GraphDB / Cypher, 선택적인
``[graph]`` extra 필요). 이 프로토콜 위쪽 코드는 어떤 백엔드를 쓰든 무관하게(backend-agnostic)
동작한다 — 즉 호출부는 GraphStore 인터페이스에만 의존하면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class GraphRecord:
    """라벨이 달린 이미지 한 장: 그 안에 어떤 객체 클래스(표시 이름)들이 들어 있는지.

    그래프 구축의 최소 입력 단위다. classes는 사람이 읽는 표시 이름(class id가 아님)으로,
    캡션과 동일한 명명 규칙을 따른다.
    """

    image_path: str
    folder: str
    # field(default_factory=list): 가변 기본값(빈 리스트)을 인스턴스마다 새로 만들어,
    # 모든 레코드가 같은 리스트를 공유하는 흔한 버그를 피한다.
    classes: list[str] = field(default_factory=list)


@runtime_checkable
class GraphStore(Protocol):
    """객체 그래프 저장소가 충족해야 하는 인터페이스(구조적 타이핑).

    @runtime_checkable 덕분에 isinstance()로 "이 객체가 GraphStore처럼 생겼는지"를
    런타임에 확인할 수 있다. 구현체는 이 클래스를 상속하지 않고 메서드만 맞추면 된다.
    """

    # 백엔드 식별용 이름(예: "memory", "kuzu"). 로깅/진단에 쓰인다.
    name: str

    def build(self, records: Sequence[GraphRecord]) -> None:
        """그래프를 처음부터 (재)구축한다."""

    def count(self) -> int:
        """그래프에 담긴, 라벨이 있는 이미지 개수."""

    def class_counts(self) -> dict[str, int]:
        """클래스 이름 -> 그 클래스를 포함한 이미지 수."""

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        """``class_name``과 함께 등장(co-occur)하는 클래스들을 공유 이미지 수 내림차순으로."""

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        """``class_names``를 "모두(EVERY)" 포함하는 이미지 경로들 (AND 조건)."""

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        """``class_names`` 중 "하나라도(ANY)" 포함하는 이미지 경로들 (OR 조건)."""

    def edges(self, top: int = 50) -> list[tuple[str, str, int]]:
        """상위 동시출현(co-occurrence) 간선들을 (클래스_a, 클래스_b, 공유 이미지 수)로."""

    def clear(self) -> None:
        """그래프를 비운다(모든 노드/간선 제거)."""
        ...

    def close(self) -> None:
        """점유 중인 자원을 해제한다(kuzu의 디렉터리 잠금 등). 메모리 백엔드는 no-op."""
