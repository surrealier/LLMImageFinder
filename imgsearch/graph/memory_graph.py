"""순수 파이썬 인메모리 객체 그래프 — 기본 백엔드(항상 사용 가능).

이미지->클래스, 클래스->이미지 인접 관계를 평범한 dict/set으로 들고 있을 뿐이지만,
이 규모(이미지 수천 장)의 데이터셋에서는 동시출현(co-occurrence), AND/OR 멤버십 질의,
간선 나열을 처리하기에 충분하다. 외부 의존성이 전혀 없고 완전히 결정적(deterministic)이라
테스트와 기본 동작에 적합하다.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Sequence

from imgsearch.graph.base import GraphRecord


class MemoryGraphStore:
    """양방향 인접 맵으로 그래프 질의를 제공하는 인메모리 GraphStore 구현."""

    # GraphStore 프로토콜이 요구하는 백엔드 식별 이름.
    name = "memory"

    def __init__(self) -> None:
        """빈 인접 맵 두 개와, 동시성 보호용 재진입 락(RLock)을 준비한다."""
        # 이미지 경로 -> 그 이미지에 들어 있는 클래스 집합.
        self._img_classes: dict[str, set[str]] = {}
        # 클래스 이름 -> 그 클래스를 포함하는 이미지 경로 집합(역인덱스).
        # 두 방향을 모두 들고 있어야 동시출현/멤버십 질의를 빠르게 처리할 수 있다.
        self._class_images: dict[str, set[str]] = {}
        # build()는 GraphBuildWorker 스레드에서 돌고, 읽기는 질의 스레드/GUI 스레드에서
        # 돈다. 락으로 직렬화해, 읽는 쪽이 "절반만 채워진 그래프"를 보는 일을 막는다.
        # RLock(재진입 락)을 쓰는 이유: 락을 쥔 메서드가 같은 스레드에서 다른 락 메서드를
        # 호출해도 데드락에 빠지지 않게 하기 위함.
        self._lock = threading.RLock()

    def build(self, records: Sequence[GraphRecord]) -> None:
        """레코드 목록으로 두 인접 맵을 새로 만들고, 한 번에 교체(swap)한다.

        맵을 제자리(in-place)에서 수정하지 않고 로컬에 완성한 뒤 통째로 바꾸는 이유는,
        구축 도중의 부분 상태가 읽기 스레드에 노출되지 않게 하기 위함이다.
        """
        # 먼저 락 밖에서 로컬 맵을 완성한다(구축은 시간이 걸릴 수 있으므로 락 점유 최소화).
        img_classes: dict[str, set[str]] = {}
        class_images: dict[str, set[str]] = {}
        for r in records:
            # 빈 문자열 클래스는 제외하고 집합으로 만든다(중복 제거 겸 빠른 멤버십 체크).
            classes = {c for c in r.classes if c}
            if not classes:
                continue  # 라벨이 없는 이미지는 그래프에 넣지 않는다.
            img_classes[r.image_path] = classes
            for c in classes:
                # 역인덱스 구축: 처음 보는 클래스면 빈 집합을 만들고 이미지 경로를 추가.
                class_images.setdefault(c, set()).add(r.image_path)
        # 완성된 맵들을 락 안에서 원자적으로 교체 → 읽기 측은 "전부 옛 것" 또는 "전부 새 것"만
        # 보게 되어(all-or-nothing) 중간 상태를 절대 보지 못한다.
        with self._lock:
            self._img_classes = img_classes
            self._class_images = class_images

    def count(self) -> int:
        """라벨이 있는 이미지 개수(= 그래프에 등록된 이미지 노드 수)."""
        with self._lock:
            return len(self._img_classes)

    def class_counts(self) -> dict[str, int]:
        """클래스별 포함 이미지 수 맵. 역인덱스 집합의 크기를 그대로 센다."""
        with self._lock:
            return {c: len(imgs) for c, imgs in self._class_images.items()}

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        """``class_name``과 같은 이미지에 함께 등장한 클래스들을 빈도 내림차순으로 상위 top개.

        방식: class_name을 가진 이미지들을 모두 돌며, 그 이미지가 가진 다른 클래스마다
        +1씩 카운트한다(자기 자신은 제외).
        """
        co: Counter = Counter()
        with self._lock:
            for img in self._class_images.get(class_name, set()):
                for c in self._img_classes.get(img, ()):  # 그 이미지에 들어 있는 클래스들
                    if c != class_name:  # 자기 자신과의 동시출현은 세지 않는다.
                        co[c] += 1
        # Counter.most_common(top): 빈도 내림차순 상위 top개를 (이름, 횟수)로 반환.
        return co.most_common(top)

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        """주어진 클래스를 "전부" 가진 이미지 경로들(AND). 결과는 경로 정렬.

        집합 교집합(intersection)으로 구현한다.
        """
        names = [c for c in class_names if c]  # 빈 문자열 제거
        if not names:
            return []
        with self._lock:
            # 각 클래스의 이미지 집합을 모은다.
            sets = [self._class_images.get(c, set()) for c in names]
            # 요청한 클래스 중 하나라도 아예 없으면(빈 집합) 교집합도 비므로 즉시 빈 결과.
            if any(not s for s in sets):  # 요청한 클래스가 그래프에 없음 -> 충족 이미지 없음
                return []
            # 모든 집합의 교집합 = 모든 클래스를 동시에 가진 이미지. 정렬로 결과 안정화.
            return sorted(set.intersection(*sets))

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        """주어진 클래스 중 "하나라도" 가진 이미지 경로들(OR). 합집합으로 구현, 정렬 반환."""
        out: set[str] = set()
        with self._lock:
            for c in class_names:
                # |= : 합집합 누적. 없는 클래스는 빈 집합이라 결과에 영향 없음.
                out |= self._class_images.get(c, set())
        return sorted(out)

    def edges(self, top: int = 50) -> list[tuple[str, str, int]]:
        """동시출현 간선을 (클래스_a, 클래스_b, 공유 이미지 수)로, 빈도순 상위 top개.

        각 이미지가 가진 클래스 집합에서 모든 (i<j) 쌍을 만들어 카운트한다.
        """
        co: Counter = Counter()
        with self._lock:
            for classes in self._img_classes.values():
                # 정렬해서 쌍의 순서를 (a,b) with a<b로 고정 → (a,b)와 (b,a)가 따로 세지지 않음.
                ordered = sorted(classes)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        co[(ordered[i], ordered[j])] += 1
        return [(a, b, n) for (a, b), n in co.most_common(top)]

    def clear(self) -> None:
        """두 인접 맵을 비워 그래프를 초기화한다(락으로 보호)."""
        with self._lock:
            self._img_classes = {}
            self._class_images = {}

    def close(self) -> None:
        """해제할 외부 자원이 없으므로 아무것도 하지 않는다(프로토콜 호환용 no-op)."""
        pass
