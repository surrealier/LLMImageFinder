"""Embedded GraphDB backend (kuzu) — real Cypher over a persisted graph.

Optional: requires the ``[graph]`` extra (``uv sync --extra graph``). The schema is
  (:Image {path})  -[:HAS_OBJECT]->  (:Class {name})
  (:Image {path})  -[:IN_FOLDER]->   (:Folder {path})
Co-occurrence is derived at query time with Cypher. If kuzu is missing or any
operation fails, the registry falls back to the pure-python MemoryGraphStore.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from imgsearch.graph.base import GraphRecord
from imgsearch.logging_setup import get_logger

log = get_logger("kuzu_graph")


class KuzuGraphStore:
    name = "kuzu"

    def __init__(self, db_dir: str | Path) -> None:
        import kuzu  # raises if the [graph] extra is not installed

        self._kuzu = kuzu
        self._db_dir = Path(db_dir)
        self._db = None
        # serialize build (DETACH DELETE + CREATE) vs reads vs close() across threads
        self._lock = threading.RLock()
        self._open()

    # -- lifecycle --
    def _open(self) -> None:
        # one Database for the process; connections are made per-operation so reads
        # from a worker thread are safe (kuzu: one Database, many Connections)
        self._db_dir.parent.mkdir(parents=True, exist_ok=True)
        self._db = self._kuzu.Database(str(self._db_dir))
        conn = self._kuzu.Connection(self._db)
        for s in (
            "CREATE NODE TABLE IF NOT EXISTS Image(path STRING, folder STRING, PRIMARY KEY (path))",
            "CREATE NODE TABLE IF NOT EXISTS Class(name STRING, PRIMARY KEY (name))",
            "CREATE NODE TABLE IF NOT EXISTS Folder(path STRING, PRIMARY KEY (path))",
            "CREATE REL TABLE IF NOT EXISTS HAS_OBJECT(FROM Image TO Class)",
            "CREATE REL TABLE IF NOT EXISTS IN_FOLDER(FROM Image TO Folder)",
        ):
            conn.execute(s)

    def _rows(self, query: str, params: dict | None = None) -> list[list]:
        with self._lock:
            if self._db is None:
                return []
            conn = self._kuzu.Connection(self._db)  # fresh connection per call
            res = conn.execute(query, parameters=params or {})
            out: list[list] = []
            while res.has_next():
                out.append(res.get_next())
            return out

    def close(self) -> None:
        # release the directory lock so the next session/process can re-open
        with self._lock:
            self._db = None

    def _wipe(self, conn) -> None:
        # in-place clear (DETACH DELETE removes nodes + their rels). Avoids deleting
        # the on-disk dir while the Database object still holds it open on Windows.
        for tbl in ("Image", "Class", "Folder"):
            conn.execute(f"MATCH (n:{tbl}) DETACH DELETE n")

    # -- build --
    def build(self, records: Sequence[GraphRecord]) -> None:
        images: list[dict] = []
        folders: set[str] = set()
        classes: set[str] = set()
        has_obj: list[dict] = []
        in_folder: list[dict] = []
        for r in records:
            cs = [c for c in dict.fromkeys(r.classes) if c]  # unique, order-stable
            if not cs:
                continue
            images.append({"path": r.image_path, "folder": r.folder})
            folders.add(r.folder)
            in_folder.append({"img": r.image_path, "folder": r.folder})
            for c in cs:
                classes.add(c)
                has_obj.append({"img": r.image_path, "cls": c})

        with self._lock:
            if self._db is None:
                self._open()
            conn = self._kuzu.Connection(self._db)
            self._wipe(conn)  # idempotent rebuild (also empties on no records)
            if not images:
                return
            conn.execute(
                "UNWIND $rows AS r CREATE (:Image {path: r.path, folder: r.folder})",
                parameters={"rows": images},
            )
            conn.execute(
                "UNWIND $rows AS r CREATE (:Folder {path: r})",
                parameters={"rows": list(folders)},
            )
            conn.execute(
                "UNWIND $rows AS r CREATE (:Class {name: r})",
                parameters={"rows": list(classes)},
            )
            conn.execute(
                "UNWIND $rows AS r MATCH (i:Image {path: r.img}), (c:Class {name: r.cls}) "
                "CREATE (i)-[:HAS_OBJECT]->(c)",
                parameters={"rows": has_obj},
            )
            conn.execute(
                "UNWIND $rows AS r MATCH (i:Image {path: r.img}), (f:Folder {path: r.folder}) "
                "CREATE (i)-[:IN_FOLDER]->(f)",
                parameters={"rows": in_folder},
            )

    # -- queries --
    def count(self) -> int:
        rows = self._rows("MATCH (i:Image) RETURN count(i)")
        return int(rows[0][0]) if rows else 0

    def class_counts(self) -> dict[str, int]:
        rows = self._rows(
            "MATCH (i:Image)-[:HAS_OBJECT]->(c:Class) RETURN c.name, count(i)"
        )
        return {str(name): int(n) for name, n in rows}

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        rows = self._rows(
            "MATCH (c:Class {name: $n})<-[:HAS_OBJECT]-(i:Image)-[:HAS_OBJECT]->(c2:Class) "
            "WHERE c2.name <> $n RETURN c2.name, count(i) AS n ORDER BY n DESC, c2.name LIMIT $top",
            {"n": class_name, "top": int(top)},
        )
        return [(str(name), int(n)) for name, n in rows]

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        names = [c for c in class_names if c]
        if not names:
            return []
        rows = self._rows(
            "MATCH (i:Image)-[:HAS_OBJECT]->(c:Class) WHERE c.name IN $names "
            "WITH i, count(DISTINCT c.name) AS m WHERE m = $k RETURN i.path ORDER BY i.path",
            {"names": names, "k": len(names)},
        )
        return [str(r[0]) for r in rows]

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        names = [c for c in class_names if c]
        if not names:
            return []
        rows = self._rows(
            "MATCH (i:Image)-[:HAS_OBJECT]->(c:Class) WHERE c.name IN $names "
            "RETURN DISTINCT i.path ORDER BY i.path",
            {"names": names},
        )
        return [str(r[0]) for r in rows]

    def edges(self, top: int = 50) -> list[tuple[str, str, int]]:
        rows = self._rows(
            "MATCH (a:Class)<-[:HAS_OBJECT]-(i:Image)-[:HAS_OBJECT]->(b:Class) "
            "WHERE a.name < b.name RETURN a.name, b.name, count(i) AS n "
            "ORDER BY n DESC, a.name, b.name LIMIT $top",
            {"top": int(top)},
        )
        return [(str(a), str(b), int(n)) for a, b, n in rows]

    def clear(self) -> None:
        with self._lock:
            if self._db is None:
                self._open()
            self._wipe(self._kuzu.Connection(self._db))
