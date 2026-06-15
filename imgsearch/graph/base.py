"""GraphStore protocol + the record fed in at build time.

Two interchangeable backends implement this: a pure-python in-memory store (always
available, the default) and an embedded kuzu store (real GraphDB / Cypher, optional
``[graph]`` extra). Everything above the protocol is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class GraphRecord:
    """One labeled image: which object classes (display names) it contains."""

    image_path: str
    folder: str
    classes: list[str] = field(default_factory=list)


@runtime_checkable
class GraphStore(Protocol):
    name: str

    def build(self, records: Sequence[GraphRecord]) -> None:
        """(Re)build the graph from scratch."""

    def count(self) -> int:
        """Number of labeled images in the graph."""

    def class_counts(self) -> dict[str, int]:
        """class name -> number of images containing it."""

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        """Classes that co-occur with ``class_name``, by shared-image count, desc."""

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        """Image paths containing EVERY one of ``class_names`` (AND)."""

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        """Image paths containing ANY of ``class_names`` (OR)."""

    def edges(self, top: int = 50) -> list[tuple[str, str, int]]:
        """Top co-occurrence edges as (class_a, class_b, shared_image_count)."""

    def clear(self) -> None:
        ...

    def close(self) -> None:
        """Release any held resources (kuzu directory lock); no-op for memory."""
