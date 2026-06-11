"""Backend Protocols. The app talks only to these; concrete impls live alongside.

Three roles:
  * Embedder  — image & text -> a shared, L2-normalized vector space (retrieval core).
  * Captioner — representative image -> a short Korean description (metadata + RAG doc).
  * ChatLLM   — refine the Korean query + write a grounded Korean summary of results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # avoid import cycle at runtime
    from imgsearch.core.models import FolderHit


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_image(self, paths: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized. Order matches ``paths``."""

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized. Order matches ``texts``."""


@runtime_checkable
class Captioner(Protocol):
    name: str

    def caption(self, image_path: str, context: str = "") -> str:
        """A short Korean caption for the image. ``context`` may carry sidecar text."""


@runtime_checkable
class ChatLLM(Protocol):
    name: str

    def refine_query(self, ko_text: str) -> str:
        """Clean/expand the Korean query into search text used for embedding."""

    def summarize(self, query: str, hits: "list[FolderHit]") -> str:
        """A short Korean answer grounded in the retrieved hits."""
