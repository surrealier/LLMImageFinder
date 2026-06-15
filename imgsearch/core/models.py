"""인덱스 파이프라인 · 서비스 계층 · UI 사이를 오가는 단순 데이터 클래스 모음.

이 모듈에는 로직이 거의 없다. 계층 간에 값을 주고받기 위한 순수 데이터 컨테이너
(dataclass)만 모아두어, 서로 다른 모듈이 dict 키 이름에 의존하지 않고 타입 안전하게
데이터를 교환하도록 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeafFolder:
    """이미지 파일이 들어 있는 말단(leaf) 디렉터리.

    이 폴더 단위로 대표 이미지 1장을 골라 검색 색인을 만든다(즉, 한 폴더를 하나의
    대표로 '접는다'). 폴더 안의 여러 이미지를 개별로 색인하지 않고 대표 1장만
    색인하는 것이 이 앱의 기본 전략이다.
    """

    path: str
    image_paths: list[str]
    # 짝이 되는 sidecar(메타데이터) 파일들에서 추출해 이어 붙인 텍스트.
    # 키워드(BM25) 검색의 보조 입력으로 쓰인다.
    sidecar_text: str = ""  # concatenated text from paired sidecar files


@dataclass
class RepImage:
    """말단 폴더에 대해 선정된 대표(representative) 이미지.

    폴더 안 여러 이미지 중 검색 색인에 실제로 넣을 1장이 무엇인지, 그리고 어떤 방식으로
    골랐는지를 담는다.
    """

    folder: str
    image_path: str
    member_count: int  # 이 대표가 대신하는 폴더 내 이미지 총 개수
    # 대표를 고른 방식(추적/디버깅용 표식):
    #   "centroid"        — 임베딩 평균(중심)에 가장 가까운 이미지
    #   "dhash+centroid"  — dHash로 중복 제거 후 중심 기준 선정
    #   "middle"          — 그냥 가운데(중간 인덱스) 이미지
    #   "only"            — 폴더에 이미지가 1장뿐
    method: str  # how it was picked: "centroid" | "dhash+centroid" | "middle" | "only"


@dataclass
class FolderHit:
    """검색 결과 한 줄: 폴더의 대표 이미지와 '왜 매칭됐는지' 정보.

    핵심 설계 포인트 — ``score``는 검색 모드와 무관하게 '항상' 코사인 유사도다.
    덕분에 UI의 점수 임계값(threshold) 필터가 어떤 모드에서든 동일한 의미를 갖는다.
    반면 ``fused_score``는 하이브리드/키워드 모드에서 '정렬 순서'를 정하는 데 쓰는
    RRF(역순위 융합) 점수이며, 순수 벡터 모드에서는 0.0이다(정렬은 score로 충분하므로).
    ``match``는 출처(provenance)를 기록한다: ""|"vector"|"keyword"|"both".
    """

    folder: str
    image_path: str
    caption: str
    score: float  # 항상 코사인 유사도. UI 임계값 필터의 일관된 기준이 된다.
    member_count: int = 0
    sidecar_text: str = ""
    fused_score: float = 0.0  # RRF 융합 점수. 하이브리드/키워드 정렬 전용(벡터 모드는 0.0)
    match: str = ""  # 출처 표식: ""|"vector"|"keyword"|"both"(+ agentic에서는 "graph")


@dataclass
class QueryResult:
    """한 번의 질의에 대한 전체 결과 묶음(원문 질의·정제 질의·히트 목록·요약)."""

    query: str  # 사용자가 입력한 원문 질의
    refined: str  # LLM/규칙으로 정제(확장)된 실제 검색에 쓰인 질의
    hits: list[FolderHit] = field(default_factory=list)
    summary: str = ""  # 선택적 자연어 요약(요약 비활성/실패 시 빈 문자열)


@dataclass
class IndexProgress:
    """색인 작업의 진행 상황. 워커 스레드 → UI 프로그레스 바로 전달된다."""

    current: int
    total: int
    # 현재 단계. UI는 이 값으로 진행 표시/완료 처리를 분기한다:
    #   "scan"(폴더 스캔) | "index"(색인 작성) | "done"(완료) | "cancelled"(취소됨)
    phase: str  # "scan" | "index" | "done" | "cancelled"
    folder: str = ""  # 현재 처리 중인 폴더(표시용)


@dataclass
class BuildReport:
    """색인 빌드 결과 요약: 몇 개를 기록·건너뜀·정리(prune)했는지.

    빌드가 끝난 뒤 UI에 결과를 보고하기 위한 집계 객체다.
    """

    n_written: int = 0  # 실제로 기록(색인)된 레코드 수
    total: int = 0  # 처리 대상 전체 수
    # 건너뛴 항목들: (경로, 오류 메시지) 튜플 목록. UI에서 실패 사유를 보여줄 때 쓴다.
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, error)
    pruned: int = 0  # 디스크에서 파일이 사라져 색인에서 제거된 레코드 수
    cancelled: bool = False  # 사용자가 중간에 취소했는지 여부
