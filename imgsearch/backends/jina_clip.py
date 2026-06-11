"""Real multilingual CLIP embedder (jina-clip-v2) via sentence-transformers.

Text and images map into one L2-normalized space, Matryoshka-truncated to ``dim``.
Korean text->image retrieval works directly (no translation). Requires the ``[embed]``
extra (``uv sync --extra embed``); if torch/sentence-transformers are missing, importing
this module raises and the registry falls back to the mock embedder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from imgsearch.logging_setup import get_logger

log = get_logger("jina_clip")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class JinaClipEmbedder:
    name = "jina-clip"

    def __init__(
        self,
        model_id: str = "jinaai/jina-clip-v2",
        device: str = "auto",
        dim: int = 512,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = int(dim)
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        log.info("Loading %s on %s (truncate_dim=%d)…", model_id, self.device, self.dim)
        self._model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=self.device,
            truncate_dim=self.dim,
        )

    def _load_pil(self, path: str):
        from PIL import Image

        try:
            with Image.open(path) as im:
                return im.convert("RGB").copy()
        except Exception:
            return Image.new("RGB", (32, 32), (0, 0, 0))

    def _fix_dim(self, vecs) -> np.ndarray:
        """Guarantee (n, self.dim) L2-normalized float32 regardless of model output width.

        If the remote model ignores the constructor's ``truncate_dim`` and returns wider
        vectors, truncate (Matryoshka-valid) and renormalize so stored ``embed_dim`` holds.
        """
        vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float32))
        if vecs.shape[1] < self.dim:
            raise ValueError(
                f"{self.name}: model produced dim {vecs.shape[1]} < requested {self.dim}"
            )
        if vecs.shape[1] > self.dim:
            vecs = vecs[:, : self.dim]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=np.float32)
        images = [self._load_pil(p) for p in paths]
        vecs = self._model.encode(
            images,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        return self._fix_dim(vecs)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=max(self.batch_size, 32),
        )
        return self._fix_dim(vecs)
