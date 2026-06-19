"""모델 없이도 앱 전체가 돌아가도록 하는 결정적(deterministic)·ML-free 백엔드 모음.

설치된 모델이 하나도 없어도 앱을 실행/테스트할 수 있게 하는 것이 목적이다(mock-first 설계).

  * MockEmbedder   — 이미지 벡터 = (경로 텍스트 특징) + 약간의 실제 시각 신호
                     (16x16 그레이스케일 + RGB 히스토그램을 고정 랜덤 투영); 텍스트
                     벡터 = 해시 기반 한국어 특징. 동일한 512차원 코사인 공간을 공유한다.
  * MockCaptioner  — 폴더/파일명 토큰, 밝기(낮/밤), 작은 개념 사전으로 한국어 캡션 생성;
                     사이드카 context도 함께 반영한다.
  * MockChatLLM    — 규칙 기반 한국어 질의 확장 + 템플릿 기반 한국어 요약.

모든 출력은 결정적이다(md5 해시 버킷 + 상수 시드로 고정한 랜덤 투영).
따라서 결과가 재현 가능하며 단위 테스트로 검증할 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from imgsearch import koutil
from imgsearch.core.models import FolderHit

# 랜덤 투영 행렬의 시드. 상수로 고정해야 실행할 때마다 같은 투영(=같은 임베딩)이 나와 재현성이 보장된다.
_PROJ_SEED = 20260609
# 저수준 시각 디스크립터의 차원: 16*16 그레이스케일(256) + 채널당 16빈 RGB 히스토그램(3*16=48) = 304.
_RAW_DIM = 304  # 16*16 grayscale (256) + 3*16 RGB histogram (48)


def _seed_from(path: str) -> int:
    """경로 문자열로부터 결정적 정수 시드를 만든다(읽기 실패 시 RNG 폴백에 사용).

    같은 경로면 항상 같은 시드 -> 같은 폴백 벡터가 나오므로 결정성이 유지된다.
    """
    import hashlib

    # md5 해시의 앞 8바이트를 little-endian 정수로 해석해 시드로 사용한다.
    return int.from_bytes(hashlib.md5(path.encode("utf-8")).digest()[:8], "little")


def _path_text(path: str) -> str:
    """파일 경로에서 검색에 쓸 텍스트(상위 2개 폴더명 + 확장자 없는 파일명)를 추출한다.

    데이터셋 폴더 구조 자체가 라벨 역할을 하는 경우가 많아, 경로를 약한 텍스트 신호로 활용한다.
    """
    p = Path(path)
    # parent.parts 중 마지막 2개 폴더만 사용(루트까지의 긴 경로 노이즈를 배제).
    folders = list(p.parent.parts)[-2:]
    return " ".join(folders + [p.stem])


def _visual_raw(path: str) -> np.ndarray:
    """304차원 저수준 시각 디스크립터를 만든다. 이미지를 읽지 못하면 결정적 폴백 벡터를 반환한다."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            # 16x16 그레이스케일을 1차원으로 펴서 256차원 강도(intensity) 특징으로 사용.
            gray = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32).reshape(-1)
            gray /= 255.0  # 0~1 범위로 정규화
            # 색상 히스토그램은 32x32 축소본에서 계산(연산량 절감, 전역 색 분포만 필요).
            small = np.asarray(im.resize((32, 32)), dtype=np.float32)
        hist = []
        for c in range(3):  # R, G, B 각 채널별 16빈 히스토그램
            h, _ = np.histogram(small[:, :, c], bins=16, range=(0, 255))
            hist.append(h.astype(np.float32))
        hist_v = np.concatenate(hist)
        s = float(hist_v.sum())
        if s > 0:
            hist_v /= s  # 픽셀 수에 무관하도록 확률 분포로 정규화
        return np.concatenate([gray, hist_v]).astype(np.float32)
    except Exception:
        # 손상/미지원 이미지는 예외를 던지지만, 임베딩은 멈추면 안 되므로 경로 시드 기반 결정적 난수로 대체.
        rng = np.random.default_rng(_seed_from(path))
        return rng.random(_RAW_DIM).astype(np.float32)


def _mean_brightness(path: str) -> float | None:
    """이미지의 평균 밝기(0~255)를 반환한다. 읽기 실패 시 None(밝기 미상)."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            # 그레이스케일 32x32 축소본의 평균값으로 전체 밝기를 근사한다.
            return float(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).mean())
    except Exception:
        return None


class MockEmbedder:
    """모델 없이 동작하는 임베더. 경로 텍스트 특징과 약한 시각 신호를 합쳐 512차원 벡터를 만든다."""

    name = "mock"

    def __init__(self, dim: int = 512) -> None:
        """``dim`` 차원의 임베더를 만들고, 시드를 고정한 랜덤 투영 행렬을 미리 생성한다."""
        self.dim = int(dim)
        # _PROJ_SEED로 고정 -> 인스턴스마다 동일한 투영 행렬 -> 재현 가능한 임베딩.
        rng = np.random.default_rng(_PROJ_SEED)
        # (304, dim) 가우시안 랜덤 투영: 저수준 시각 특징을 임베딩 공간으로 사상한다.
        self._proj = rng.standard_normal((_RAW_DIM, self.dim)).astype(np.float32)

    def _embed_one_image(self, path: str) -> np.ndarray:
        """이미지 1장을 임베딩한다: 경로 텍스트 특징을 주(主) 신호로, 시각 특징을 보조 신호로 결합."""
        # 텍스트 특징을 주 신호로 둬서 텍스트 질의와 같은 공간에 정렬되도록 한다(교차 검색 가능).
        tfeat = koutil.text_features(_path_text(path), self.dim)
        # 저수준 시각 특징을 랜덤 투영으로 dim 차원에 사상.
        v = _visual_raw(path) @ self._proj
        nv = float(np.linalg.norm(v))
        if nv > 0:
            v = v / nv  # 시각 벡터를 단위 길이로 정규화한 뒤
        # 0.25 가중치로 약하게만 섞는다(텍스트 신호 우위를 유지하면서 시각 정보로 약간의 변별력 추가).
        combo = tfeat + 0.25 * v.astype(np.float32)
        c = float(np.linalg.norm(combo))
        if c > 0:
            combo = combo / c  # 최종 벡터를 L2 정규화 -> 코사인 유사도 = 내적이 되도록
        return combo.astype(np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        """이미지 경로 시퀀스를 (n, dim) 정규화 벡터 행렬로 임베딩한다."""
        if not paths:
            # 빈 입력은 vstack이 실패하므로 모양이 맞는 (0, dim) 빈 배열을 명시적으로 반환.
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one_image(p) for p in paths])

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """텍스트 시퀀스를 (n, dim) 정규화 벡터 행렬로 임베딩한다(이미지와 동일 공간)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 텍스트는 시각 신호 없이 한국어 해시 특징만으로 임베딩한다.
        return np.vstack([koutil.text_features(t, self.dim) for t in texts]).astype(np.float32)


class MockCaptioner:
    """모델 없이 규칙 기반으로 한국어 캡션을 만드는 캡셔너(밝기 + 경로/개념 힌트 조합)."""

    name = "mock"

    def caption(self, image_path: str, context: str = "") -> str:
        """이미지 밝기와 경로/메모에서 추출한 개념 힌트를 조합해 짧은 한국어 캡션을 만든다."""
        p = Path(image_path)
        bright = _mean_brightness(image_path)
        # 평균 밝기를 구간으로 나눠 조명/시간대를 자연어로 표현한다(임계값은 경험적으로 정한 값).
        if bright is None:
            light = "unknown-lighting"  # 이미지를 읽지 못한 경우
        elif bright < 60:
            light = "dark (night)"
        elif bright < 110:
            light = "dimly lit"
        elif bright < 185:
            light = "normally lit"
        else:
            light = "bright (daytime)"

        # 경로 텍스트와 context를 합쳐 개념 사전에서 매칭되는 힌트(예: 차량, 화재 등)를 뽑는다.
        hints = koutil.concept_hints(_path_text(image_path) + " " + context)
        caption = f"{light} photo"
        if hints:
            caption += " — " + ", ".join(hints)
        caption += f" (folder: {p.parent.name})"
        # context의 연속 공백을 단일 공백으로 정리한 뒤
        ctx = " ".join(context.split())
        if ctx:
            caption += f". note: {ctx[:80]}"  # 메모는 너무 길어지지 않도록 80자로 자른다
        return caption


class MockChatLLM:
    """모델 없이 동작하는 대화형 LLM. 규칙 기반 질의 확장과 템플릿 기반 요약을 제공한다."""

    name = "mock"

    def refine_query(self, ko_text: str) -> str:
        """규칙 기반 한국어 질의 확장(koutil.expand_query에 위임)."""
        return koutil.expand_query(ko_text)

    def summarize(self, query: str, hits: list[FolderHit]) -> str:
        """검색 결과(hits)를 템플릿에 끼워 한국어 요약 문장을 만든다. 결과가 없으면 안내 문구 반환."""
        if not hits:
            return (
                f"No images matched '{query}'. "
                "Try different keywords or phrasing."
            )
        # 상위 3개 폴더명만 미리보기로 보여준다.
        tops = ", ".join(Path(h.folder).name for h in hits[:3])
        return (
            f"Found {len(hits)} matching folders for '{query}'. "
            f"Most similar: {tops}. Top caption: {hits[0].caption}"
        )
