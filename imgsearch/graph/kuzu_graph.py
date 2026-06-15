"""임베디드 GraphDB 백엔드(kuzu) — 영속화된 그래프 위에서 진짜 Cypher를 실행한다.

선택적 백엔드: ``[graph]`` extra가 필요하다(``uv sync --extra graph``). 스키마는 다음과 같다.
  (:Image {path})  -[:HAS_OBJECT]->  (:Class {name})
  (:Image {path})  -[:IN_FOLDER]->   (:Folder {path})
동시출현(co-occurrence)은 미리 저장하지 않고 질의 시점에 Cypher로 유도한다. kuzu가
없거나 어떤 연산이 실패하면, 레지스트리가 순수 파이썬 MemoryGraphStore로 대체(fallback)한다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from imgsearch.graph.base import GraphRecord
from imgsearch.logging_setup import get_logger

log = get_logger("kuzu_graph")


class KuzuGraphStore:
    """kuzu 임베디드 그래프 DB를 GraphStore 인터페이스로 감싼 구현."""

    name = "kuzu"

    def __init__(self, db_dir: str | Path) -> None:
        """kuzu를 지연 임포트해 DB를 열고 스키마를 준비한다.

        kuzu import가 실패하면(=extra 미설치) 여기서 예외가 나고, 레지스트리가 이를
        잡아 메모리 백엔드로 대체한다.
        """
        import kuzu  # [graph] extra가 설치돼 있지 않으면 여기서 ImportError가 난다

        self._kuzu = kuzu
        self._db_dir = Path(db_dir)
        self._db = None  # _open()에서 실제 Database 객체로 채워진다. close() 후엔 다시 None.
        # build(DETACH DELETE + CREATE), 읽기, close()를 스레드 간 직렬화한다.
        # RLock인 이유: 같은 스레드에서 락을 쥔 채 다른 락 메서드를 호출해도 안전하도록.
        self._lock = threading.RLock()
        self._open()

    # -- 생명주기(lifecycle) --
    def _open(self) -> None:
        """DB 디렉터리를 만들고 Database를 연 뒤, 노드/관계 테이블 스키마를 보장한다.

        프로세스당 Database는 하나만 두고, 연결(Connection)은 연산마다 새로 만든다.
        kuzu의 동시성 모델이 "Database 하나, Connection 다수"라서, 워커 스레드에서의
        읽기를 안전하게 하려면 이 패턴을 따라야 한다.
        """
        # 한 프로세스 = 하나의 Database. 연결은 연산별로 만든다(아래 _rows/build 참고).
        self._db_dir.parent.mkdir(parents=True, exist_ok=True)
        self._db = self._kuzu.Database(str(self._db_dir))
        conn = self._kuzu.Connection(self._db)
        # IF NOT EXISTS로 멱등하게 스키마 생성 — 이미 있으면 무시되어 재실행해도 안전하다.
        for s in (
            "CREATE NODE TABLE IF NOT EXISTS Image(path STRING, folder STRING, PRIMARY KEY (path))",
            "CREATE NODE TABLE IF NOT EXISTS Class(name STRING, PRIMARY KEY (name))",
            "CREATE NODE TABLE IF NOT EXISTS Folder(path STRING, PRIMARY KEY (path))",
            "CREATE REL TABLE IF NOT EXISTS HAS_OBJECT(FROM Image TO Class)",
            "CREATE REL TABLE IF NOT EXISTS IN_FOLDER(FROM Image TO Folder)",
        ):
            conn.execute(s)

    def _rows(self, query: str, params: dict | None = None) -> list[list]:
        """Cypher 질의를 실행하고 결과를 행(row) 리스트로 모아 반환하는 공통 헬퍼.

        모든 읽기 메서드가 이 헬퍼를 거치므로, 락/연결 생성/결과 수집 로직을 한 곳에 둔다.
        """
        with self._lock:
            # close() 이후(또는 미오픈)면 DB가 없으므로 빈 결과로 안전하게 단락.
            if self._db is None:
                return []
            conn = self._kuzu.Connection(self._db)  # 호출마다 새 연결(스레드 안전성 확보)
            res = conn.execute(query, parameters=params or {})
            out: list[list] = []
            # 결과 스트림을 끝까지 순회하며 한 행씩 모은다.
            while res.has_next():
                out.append(res.get_next())
            return out

    def close(self) -> None:
        """Database 참조를 버려 디렉터리 잠금을 푼다 → 다음 세션/프로세스가 다시 열 수 있게.

        파일을 지우는 게 아니라 참조만 None으로 만들어 GC가 핸들을 닫게 한다.
        """
        with self._lock:
            self._db = None

    def _wipe(self, conn) -> None:
        """제자리(in-place)에서 모든 노드/관계를 삭제해 그래프를 비운다.

        DETACH DELETE는 노드와 그에 붙은 관계를 함께 지운다. 디스크 디렉터리 자체를
        삭제하지 않는 이유: 윈도우에서는 Database 객체가 그 디렉터리를 연 채(잠금) 들고
        있어, 디렉터리를 지우려 하면 충돌하기 때문이다. 그래서 내용만 비운다.
        """
        for tbl in ("Image", "Class", "Folder"):
            conn.execute(f"MATCH (n:{tbl}) DETACH DELETE n")

    # -- 구축(build) --
    def build(self, records: Sequence[GraphRecord]) -> None:
        """레코드로부터 그래프를 멱등하게 재구축한다(기존 내용 비우고 새로 적재).

        먼저 노드/관계용 파라미터 배열을 메모리에서 준비한 뒤, UNWIND로 일괄 삽입한다.
        한 건씩 INSERT 하는 대신 배열을 풀어 넣어(UNWIND) 라운드트립을 크게 줄인다.
        """
        # 각 테이블에 넣을 행들을 미리 모은다(노드 먼저, 관계는 나중에 MATCH로 연결).
        images: list[dict] = []
        folders: set[str] = set()    # set: 폴더 노드 중복 생성 방지
        classes: set[str] = set()    # set: 클래스 노드 중복 생성 방지
        has_obj: list[dict] = []     # (이미지, 클래스) HAS_OBJECT 관계용
        in_folder: list[dict] = []   # (이미지, 폴더) IN_FOLDER 관계용
        for r in records:
            # dict.fromkeys: 순서를 보존하면서 중복 클래스를 제거(빈 문자열도 걸러냄).
            cs = [c for c in dict.fromkeys(r.classes) if c]  # 중복 제거, 입력 순서 유지
            if not cs:
                continue  # 클래스가 없는 이미지는 그래프에 넣지 않는다.
            images.append({"path": r.image_path, "folder": r.folder})
            folders.add(r.folder)
            in_folder.append({"img": r.image_path, "folder": r.folder})
            for c in cs:
                classes.add(c)
                has_obj.append({"img": r.image_path, "cls": c})

        with self._lock:
            # close() 이후 build가 호출될 수도 있으니 DB가 없으면 다시 연다.
            if self._db is None:
                self._open()
            conn = self._kuzu.Connection(self._db)
            self._wipe(conn)  # 멱등 재구축: 기존 데이터 제거(레코드가 없으면 그냥 비우고 끝).
            if not images:
                return  # 넣을 게 없으면 _wipe로 비운 상태 그대로 종료.
            # 1) 먼저 노드들을 만든다(Image / Folder / Class). 관계는 노드가 있어야 연결 가능.
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
            # 2) 노드가 다 만들어졌으니 MATCH로 양 끝을 찾아 관계를 잇는다.
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

    # -- 질의(queries) --
    def count(self) -> int:
        """Image 노드 개수. 결과가 비면 0."""
        rows = self._rows("MATCH (i:Image) RETURN count(i)")
        # rows는 [[count]] 형태 → 첫 행의 첫 칼럼이 개수.
        return int(rows[0][0]) if rows else 0

    def class_counts(self) -> dict[str, int]:
        """클래스 이름 -> 그 클래스를 가리키는 이미지 수. HAS_OBJECT를 집계한다."""
        rows = self._rows(
            "MATCH (i:Image)-[:HAS_OBJECT]->(c:Class) RETURN c.name, count(i)"
        )
        return {str(name): int(n) for name, n in rows}

    def cooccurring(self, class_name: str, top: int = 20) -> list[tuple[str, int]]:
        """``class_name``과 같은 이미지를 공유하는 다른 클래스들을 빈도 내림차순 상위 top개.

        Cypher 패턴: class_name 노드에서 그것을 가진 이미지(i)로 거슬러 올라가, 그 이미지가
        가리키는 또 다른 클래스(c2)로 내려간다. WHERE로 자기 자신(c2 == class_name)은 제외.
        파라미터 바인딩($n, $top)을 써서 Cypher 인젝션을 방지한다.
        """
        rows = self._rows(
            "MATCH (c:Class {name: $n})<-[:HAS_OBJECT]-(i:Image)-[:HAS_OBJECT]->(c2:Class) "
            "WHERE c2.name <> $n RETURN c2.name, count(i) AS n ORDER BY n DESC, c2.name LIMIT $top",
            {"n": class_name, "top": int(top)},
        )
        return [(str(name), int(n)) for name, n in rows]

    def images_with_all(self, class_names: Sequence[str]) -> list[str]:
        """주어진 클래스를 "전부" 가진 이미지 경로들(AND).

        구현 트릭: IN으로 요청 클래스에 해당하는 HAS_OBJECT만 추린 뒤, 이미지별로 매칭된
        "고유 클래스 수(m)"를 세고, 그 수가 요청 개수(k)와 같은 이미지만 남긴다.
        count(DISTINCT ...)를 쓰는 이유는 같은 클래스가 중복돼도 한 번만 세기 위함이다.
        """
        names = [c for c in class_names if c]  # 빈 문자열 제거
        if not names:
            return []
        rows = self._rows(
            "MATCH (i:Image)-[:HAS_OBJECT]->(c:Class) WHERE c.name IN $names "
            "WITH i, count(DISTINCT c.name) AS m WHERE m = $k RETURN i.path ORDER BY i.path",
            {"names": names, "k": len(names)},
        )
        return [str(r[0]) for r in rows]

    def images_with_any(self, class_names: Sequence[str]) -> list[str]:
        """주어진 클래스 중 "하나라도" 가진 이미지 경로들(OR). DISTINCT로 중복 경로 제거."""
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
        """동시출현 간선 (클래스_a, 클래스_b, 공유 이미지 수)를 빈도순 상위 top개.

        a.name < b.name 조건으로 각 쌍을 한 방향으로만 세어((a,b)와 (b,a) 중복 방지)
        결정적인 결과를 만든다. ORDER BY에 이름까지 넣어 동점일 때 순서를 안정화한다.
        """
        rows = self._rows(
            "MATCH (a:Class)<-[:HAS_OBJECT]-(i:Image)-[:HAS_OBJECT]->(b:Class) "
            "WHERE a.name < b.name RETURN a.name, b.name, count(i) AS n "
            "ORDER BY n DESC, a.name, b.name LIMIT $top",
            {"top": int(top)},
        )
        return [(str(a), str(b), int(n)) for a, b, n in rows]

    def clear(self) -> None:
        """그래프 내용을 비운다. close() 이후라면 다시 열고 나서 비운다."""
        with self._lock:
            if self._db is None:
                self._open()
            self._wipe(self._kuzu.Connection(self._db))
