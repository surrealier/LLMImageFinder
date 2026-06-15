"""sentence-transformers를 통해 동작하는 실제 다국어 CLIP 임베더(jina-clip-v2).

텍스트와 이미지가 하나의 L2 정규화 공간으로 매핑되며, Matryoshka 표현 학습 덕에 ``dim``으로
잘라내도(truncate) 유효한 벡터가 된다. 한국어 텍스트->이미지 검색이 번역 없이 바로 동작한다.
``[embed]`` extra가 필요하다(``uv sync --extra embed``). torch/sentence-transformers가
없으면 이 모듈 import 시 예외가 발생하고, 레지스트리는 mock 임베더로 자동 폴백한다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from imgsearch.logging_setup import get_logger

log = get_logger("jina_clip")


def _resolve_device(device: str) -> str:
    """실행 디바이스를 결정한다. "auto"면 CUDA 사용 가능 시 "cuda", 아니면 "cpu"로 정한다."""
    if device != "auto":
        return device  # 사용자가 명시한 값("cuda"/"cpu" 등)은 그대로 존중
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        # torch가 없거나 import 자체가 실패해도 안전하게 CPU로 폴백.
        return "cpu"


class JinaClipEmbedder:
    """jina-clip-v2 모델을 로드해 이미지/텍스트를 동일 공간으로 임베딩하는 실제 임베더."""

    name = "jina-clip"

    def __init__(
        self,
        model_id: str = "jinaai/jina-clip-v2",
        device: str = "auto",
        dim: int = 512,
        batch_size: int = 16,
    ) -> None:
        """모델을 로드한다. torch/sentence-transformers 미설치 시 import에서 예외가 나며 레지스트리가 mock으로 폴백한다."""
        # 무거운 의존성은 생성자 안에서 지연 import한다 -> 모듈 import만으로는 torch를 끌어오지 않아 폴백이 깔끔해진다.
        from sentence_transformers import SentenceTransformer

        self.dim = int(dim)
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        log.info("Loading %s on %s (truncate_dim=%d)…", model_id, self.device, self.dim)
        self._model = SentenceTransformer(
            model_id,
            trust_remote_code=True,  # jina-clip은 커스텀 모델 코드를 포함하므로 원격 코드 실행 허용 필요
            device=self.device,
            truncate_dim=self.dim,  # Matryoshka 절단: 모델 측에 원하는 출력 차원을 요청
        )

    def _load_pil(self, path: str):
        """이미지를 RGB PIL 이미지로 로드한다. 읽기 실패 시 검은 32x32 더미 이미지로 대체한다."""
        from PIL import Image

        try:
            # with 블록이 닫히면 파일 핸들이 해제되므로, 모델에 넘길 사본을 .copy()로 만들어 둔다.
            with Image.open(path) as im:
                return im.convert("RGB").copy()
        except Exception:
            # 손상 이미지로 배치 전체가 실패하지 않도록, 모양만 유효한 더미를 반환(순서 보존이 중요).
            return Image.new("RGB", (32, 32), (0, 0, 0))

    def _fix_dim(self, vecs) -> np.ndarray:
        """모델 출력 폭과 무관하게 (n, self.dim) L2 정규화 float32를 보장한다.

        원격 모델이 생성자의 ``truncate_dim``을 무시하고 더 넓은 벡터를 돌려주는 경우가 있다.
        그럴 때 앞쪽을 잘라(Matryoshka 특성상 유효함) 재정규화하여, 저장된 ``embed_dim``과 항상 일치시킨다.
        """
        vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float32))
        if vecs.shape[1] < self.dim:
            # 요청한 차원보다 작게 나오면 복구 불가 -> 명시적 에러(인덱스의 차원 불일치를 조기에 차단).
            raise ValueError(
                f"{self.name}: model produced dim {vecs.shape[1]} < requested {self.dim}"
            )
        if vecs.shape[1] > self.dim:
            vecs = vecs[:, : self.dim]  # 앞쪽 dim개만 취함(Matryoshka라 의미 보존)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 0 벡터를 1로 치환해 0 나누기를 방지
        return (vecs / norms).astype(np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        """이미지 경로들을 배치 단위로 임베딩해 (n, dim) 정규화 행렬을 반환한다."""
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        images = [self._load_pil(p) for p in paths]
        vecs = self._model.encode(
            images,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        # 모델이 truncate_dim을 무시했을 수 있으므로, 차원/정규화를 한 번 더 강제한다.
        return self._fix_dim(vecs)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """텍스트들을 배치 단위로 임베딩해 (n, dim) 정규화 행렬을 반환한다(이미지와 동일 공간)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            # 텍스트는 이미지보다 가벼우므로 배치를 키워(최소 32) 처리량을 높인다.
            batch_size=max(self.batch_size, 32),
        )
        return self._fix_dim(vecs)
