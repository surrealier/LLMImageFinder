"""Plain dataclasses passed between the index pipeline, services, and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LeafFolder:
    """A directory containing image files (the unit we collapse to one representative)."""

    path: str
    image_paths: list[str]
    sidecar_text: str = ""  # concatenated text from paired sidecar files


@dataclass
class RepImage:
    """The chosen representative image for a leaf folder."""

    folder: str
    image_path: str
    member_count: int
    method: str  # how it was picked: "centroid" | "dhash+centroid" | "middle" | "only"


@dataclass
class FolderHit:
    """One search result row: a folder's representative image + why it matched.

    ``score`` is ALWAYS the cosine similarity (so the score-threshold UI filter means
    the same thing in every search mode). ``fused_score`` is the reciprocal-rank-fusion
    score used for ORDERING in hybrid/keyword mode (0.0 in pure vector mode). ``match``
    records provenance: ""|"vector"|"keyword"|"both".
    """

    folder: str
    image_path: str
    caption: str
    score: float
    member_count: int = 0
    sidecar_text: str = ""
    fused_score: float = 0.0
    match: str = ""


@dataclass
class QueryResult:
    query: str
    refined: str
    hits: list[FolderHit] = field(default_factory=list)
    summary: str = ""


@dataclass
class IndexProgress:
    current: int
    total: int
    phase: str  # "scan" | "index" | "done" | "cancelled"
    folder: str = ""


@dataclass
class BuildReport:
    """Outcome of an index build: how much was written, skipped, and pruned."""

    n_written: int = 0
    total: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, error)
    pruned: int = 0  # records whose files vanished from disk
    cancelled: bool = False
