"""Deterministic, ML-free backends so the whole app runs with zero models installed.

  * MockEmbedder   — image vector = (path-text features) + a little real visual signal
                     (16x16 grayscale + RGB histogram, fixed random projection); text
                     vector = hashed Korean features. Same 512-d cosine space.
  * MockCaptioner  — Korean caption from folder/filename tokens, brightness (낮/밤),
                     and a small concept lexicon; folds in sidecar context.
  * MockChatLLM    — rule-based Korean query expansion + a templated Korean summary.

All outputs are deterministic (md5-hashed buckets + a constant-seeded projection),
so results are reproducible and unit-testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from imgsearch import koutil
from imgsearch.core.models import FolderHit

_PROJ_SEED = 20260609
_RAW_DIM = 304  # 16*16 grayscale (256) + 3*16 RGB histogram (48)


def _seed_from(path: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.md5(path.encode("utf-8")).digest()[:8], "little")


def _path_text(path: str) -> str:
    p = Path(path)
    folders = list(p.parent.parts)[-2:]
    return " ".join(folders + [p.stem])


def _visual_raw(path: str) -> np.ndarray:
    """A 304-d low-level visual descriptor; deterministic fallback if unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            gray = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32).reshape(-1)
            gray /= 255.0
            small = np.asarray(im.resize((32, 32)), dtype=np.float32)
        hist = []
        for c in range(3):
            h, _ = np.histogram(small[:, :, c], bins=16, range=(0, 255))
            hist.append(h.astype(np.float32))
        hist_v = np.concatenate(hist)
        s = float(hist_v.sum())
        if s > 0:
            hist_v /= s
        return np.concatenate([gray, hist_v]).astype(np.float32)
    except Exception:
        rng = np.random.default_rng(_seed_from(path))
        return rng.random(_RAW_DIM).astype(np.float32)


def _mean_brightness(path: str) -> float | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return float(np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).mean())
    except Exception:
        return None


class MockEmbedder:
    name = "mock"

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)
        rng = np.random.default_rng(_PROJ_SEED)
        self._proj = rng.standard_normal((_RAW_DIM, self.dim)).astype(np.float32)

    def _embed_one_image(self, path: str) -> np.ndarray:
        tfeat = koutil.text_features(_path_text(path), self.dim)
        v = _visual_raw(path) @ self._proj
        nv = float(np.linalg.norm(v))
        if nv > 0:
            v = v / nv
        combo = tfeat + 0.25 * v.astype(np.float32)
        c = float(np.linalg.norm(combo))
        if c > 0:
            combo = combo / c
        return combo.astype(np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one_image(p) for p in paths])

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([koutil.text_features(t, self.dim) for t in texts]).astype(np.float32)


class MockCaptioner:
    name = "mock"

    def caption(self, image_path: str, context: str = "") -> str:
        p = Path(image_path)
        bright = _mean_brightness(image_path)
        if bright is None:
            light = "환경 미상의"
        elif bright < 60:
            light = "어두운(야간) 환경의"
        elif bright < 110:
            light = "다소 어두운 환경의"
        elif bright < 185:
            light = "보통 밝기의"
        else:
            light = "밝은(주간) 환경의"

        hints = koutil.concept_hints(_path_text(image_path) + " " + context)
        caption = f"{light} 사진"
        if hints:
            caption += " — " + ", ".join(hints)
        caption += f" (폴더: {p.parent.name})"
        ctx = " ".join(context.split())
        if ctx:
            caption += f". 메모: {ctx[:80]}"
        return caption


class MockChatLLM:
    name = "mock"

    def refine_query(self, ko_text: str) -> str:
        return koutil.expand_query(ko_text)

    def summarize(self, query: str, hits: list[FolderHit]) -> str:
        if not hits:
            return (
                f"'{query}'에 해당하는 이미지를 찾지 못했습니다. "
                "다른 키워드나 표현으로 다시 검색해 보세요."
            )
        tops = ", ".join(Path(h.folder).name for h in hits[:3])
        return (
            f"'{query}' 검색 결과 상위 {len(hits)}개 폴더를 찾았습니다. "
            f"가장 유사한 폴더: {tops}. 대표 설명: {hits[0].caption}"
        )
