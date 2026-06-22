"""ChromaDB 영속화 계층: 리프(leaf) 폴더 하나당 코사인 공간 레코드 하나를 저장한다.

임베딩은 항상 우리 쪽(CLIP/mock)에서 직접 만들어 넘기므로, Chroma의 내장 임베딩
함수(embedding function)는 절대 붙이지 않는다. ID는 리프 폴더 경로의 안정적인 해시값이라,
재색인(re-indexing)을 해도 같은 ID로 다시 들어가 ``upsert``가 멱등(idempotent)하게 동작한다.
즉 같은 폴더를 여러 번 색인해도 레코드가 중복 생성되지 않고 덮어써진다.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from imgsearch.core.models import FolderHit
from imgsearch.logging_setup import get_logger

log = get_logger("chroma_store")

# 기본 컬렉션 이름. 폴더별 "대표(representative)" 레코드를 담는다는 의미.
COLLECTION = "representatives"
# upsert/get/delete를 이 크기 단위로 잘라 보낸다. Chroma 한 번의 요청에 수만 건을
# 통째로 넘기면 메모리/요청 한계에 걸릴 수 있어, 배치로 나눠 안정성을 확보한다.
_UPSERT_BATCH = 256


def folder_id(folder: str) -> str:
    """폴더 경로로부터 안정적이고 결정적인(deterministic) 레코드 ID를 만든다.

    경로를 정규화한 뒤 SHA-1 해시를 ID로 사용하므로, 같은 폴더는 OS/구분자와
    무관하게 항상 같은 ID가 나온다. 덕분에 재색인 시 ``upsert``가 멱등하게 된다.
    """
    # 윈도우 역슬래시(\\)를 슬래시로 통일하고, 끝의 슬래시를 제거하고, 소문자로 낮춰
    # "같은 폴더는 표기가 달라도 같은 ID"가 되도록 한다(대소문자·구분자 차이 흡수).
    norm = str(Path(folder)).replace("\\", "/").rstrip("/").lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class ChromaStore:
    """디스크에 영속화된 ChromaDB 컬렉션을 감싼 얇은 래퍼.

    벡터 검색(쓰기/읽기)에 필요한 연산만 노출하며, 임베딩은 외부에서 주입받는다는
    설계 원칙을 강제한다(내장 임베딩 함수 미사용).
    """

    def __init__(self, persist_dir: str | Path, collection: str = COLLECTION) -> None:
        """영속 클라이언트를 열고 코사인 공간 컬렉션을 준비한다.

        chromadb는 선택적/무거운 의존성이므로 함수 안에서 지연 임포트한다(앱 기동 시
        불필요하게 로딩되지 않도록).
        """
        import chromadb
        from chromadb.config import Settings

        # 영속 디렉터리가 없으면 만들어 둔다(중간 경로 포함). 이미 있으면 무시.
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            # anonymized_telemetry=False: 외부 텔레메트리 전송 차단(오프라인/사내 환경 고려).
            # allow_reset=True: clear()에서 컬렉션을 삭제·재생성할 수 있도록 허용.
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._name = collection
        # "hnsw:space": "cosine" — 거리 척도를 코사인으로 고정. query 결과의 distance를
        # 1.0 - distance로 코사인 유사도로 환산하는 로직이 이 설정에 의존한다.
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        # 쓰기가 일어날 때마다 1씩 증가시키는 내부 리비전 번호.
        # 레코드 "개수(count)"가 그대로여도(예: refresh_captions가 문서만 다시 쓰는 경우)
        # 캐시된 어휘(lexical/BM25) 인덱스가 "변경됨"을 감지할 수 있게 하는 용도다.
        self._rev = 0

    def revision(self) -> tuple[int, int]:
        """(레코드 개수, 쓰기 리비전) 튜플 — 캐시 무효화 판단용 저렴한 키.

        count만으로는 "문서 내용만 바뀐 변경"을 못 잡으므로 _rev를 함께 묶어,
        둘 중 하나라도 달라지면 캐시를 다시 빌드하도록 한다.
        """
        return (self.count(), self._rev)

    # --- 쓰기(writes) ---
    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """레코드를 삽입하거나(있으면) 덮어쓴다. ID가 같으면 갱신되므로 멱등하다.

        임베딩/문서/메타데이터는 ids와 같은 순서로 정렬되어 있다고 가정한다.
        """
        # 빈 입력은 Chroma 호출 자체를 피한다(불필요한 왕복·리비전 증가 방지).
        if len(ids) == 0:
            return
        # numpy 배열이면 float32로 통일 후 파이썬 리스트로 변환(Chroma가 받는 형식).
        # float64를 그대로 넘기면 저장 정밀도/일관성이 어긋날 수 있어 float32로 맞춘다.
        if isinstance(embeddings, np.ndarray):
            emb_list = embeddings.astype(np.float32).tolist()
        else:
            # 이미 리스트류면 각 원소를 float로 캐스팅해 형 일관성을 보장.
            emb_list = [list(map(float, e)) for e in embeddings]
        # _UPSERT_BATCH 단위로 잘라 보낸다(대량 색인 시 단일 요청 과부하 방지).
        for i in range(0, len(ids), _UPSERT_BATCH):
            j = i + _UPSERT_BATCH
            self._col.upsert(
                ids=list(ids[i:j]),
                embeddings=emb_list[i:j],
                documents=list(documents[i:j]),
                metadatas=list(metadatas[i:j]),
            )
        # 쓰기가 실제로 일어났으므로 리비전을 올려 캐시 무효화 신호를 남긴다.
        self._rev += 1

    def update_meta(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """기존(EXISTING) 레코드의 문서+메타데이터만 교체하고, 임베딩은 그대로 보존한다.

        핵심 함정: 저장돼 있던 임베딩을 일부러 다시 읽어 명시적으로 되돌려 넘긴다.
        ``embeddings`` 없이 ``documents``만 update 하면 Chroma가 자신의 기본 임베딩
        함수로 텍스트를 "재임베딩(re-embed)"해 버려, 우리가 넣어둔 CLIP 벡터 공간이
        오염된다. 이를 막기 위한 의도적인 우회 처리다.
        """
        if len(ids) == 0:
            return
        for i in range(0, len(ids), _UPSERT_BATCH):
            j = i + _UPSERT_BATCH
            batch_ids = list(ids[i:j])
            # 먼저 이 배치의 기존 임베딩을 읽어온다(보존해서 되넣기 위함).
            got = self._col.get(ids=batch_ids, include=["embeddings"])
            # 반환된 (id, 임베딩) 쌍을 id->벡터 맵으로 정리.
            emb_by_id = {
                gid: emb for gid, emb in zip(got.get("ids", []), got.get("embeddings", []))
            }
            # 실제로 존재했던 id만 갱신 대상으로 유지한다(디스크에서 사라졌거나 처음 보는
            # id는 보존할 임베딩이 없으므로 건너뛴다). keep에는 배치 내 인덱스를 담는다.
            keep = [k for k, gid in enumerate(batch_ids) if gid in emb_by_id]
            if not keep:
                continue
            self._col.update(
                ids=[batch_ids[k] for k in keep],
                # 읽어온 임베딩을 float32 리스트로 변환해 그대로 되돌려 넣어 재임베딩을 막는다.
                embeddings=[np.asarray(emb_by_id[batch_ids[k]], dtype=np.float32).tolist() for k in keep],
                # documents/metadatas는 호출자가 넘긴 전체 시퀀스에서 (배치 시작 i + 내부 k)로 인덱싱.
                documents=[documents[i + k] for k in keep],
                metadatas=[metadatas[i + k] for k in keep],
            )
        self._rev += 1

    def delete(self, ids: Sequence[str]) -> None:
        """id로 레코드를 제거한다(예: 디스크에서 사라진 폴더/파일 정리용)."""
        ids = list(ids)
        if not ids:
            return
        # 삭제도 배치 단위로 나눠 한 번에 너무 많은 id를 넘기지 않는다.
        for i in range(0, len(ids), _UPSERT_BATCH):
            self._col.delete(ids=ids[i : i + _UPSERT_BATCH])
        self._rev += 1

    def clear(self) -> None:
        """컬렉션을 통째로 삭제했다가 다시 만든다(전체 재구축).

        부분 삭제보다 깔끔하게 비우는 방법이며, 색인 전면 재생성 시 사용한다.
        """
        try:
            self._client.delete_collection(self._name)
        except Exception:
            # 컬렉션이 애초에 없거나 삭제가 실패해도 무시하고 재생성으로 넘어간다
            # (어차피 get_or_create가 빈 컬렉션을 보장하므로 결과는 동일).
            pass
        # 동일한 코사인 설정으로 빈 컬렉션을 재생성. 설정이 어긋나면 검색 결과가
        # 달라지므로 생성 시 metadata를 처음과 똑같이 맞춘다.
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        self._rev += 1

    # --- 읽기(reads) ---
    def count(self) -> int:
        """저장된 레코드 개수. 실패 시 0(검색을 빈 결과로 안전하게 단락시키기 위함)."""
        try:
            return int(self._col.count())
        except Exception as e:
            # 폴백(0)은 유지하되, 손상/잠금 등으로 '인덱스 비어 있음'처럼 보이는 원인을 로그로 남긴다.
            log.warning("count() failed, treating store as empty: %s", e)
            return 0

    def all_ids(self) -> list[str]:
        """모든 레코드의 id 목록. include=[]로 본문 없이 id만 가볍게 가져온다.

        실패하면 빈 리스트를 반환해 호출부가 예외 없이 진행되게 한다.
        """
        try:
            return list(self._col.get(include=[]).get("ids", []))
        except Exception as e:
            log.warning("all_ids() failed, returning empty list: %s", e)
            return []

    def get_embedding(self, record_id: str) -> Optional[np.ndarray]:
        """레코드 하나의 저장된 임베딩을 반환(없으면 None).

        "예시 이미지로 검색(query-by-example)"에서 사용한다. 이미 저장된 벡터를
        재사용하므로 모델을 다시 호출(인코딩)할 필요가 없다.
        """
        try:
            got = self._col.get(ids=[record_id], include=["embeddings"])
        except Exception as e:
            log.warning("get_embedding(%s) failed: %s", record_id, e)
            return None
        embs = got.get("embeddings")
        # 결과가 없거나 비어 있으면 None. (단일 조회라 len 체크로 충분)
        if embs is None or len(embs) == 0:
            return None
        return np.asarray(embs[0], dtype=np.float32)

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        """여러 id의 저장 임베딩을 한 번의 배치 왕복으로 가져온다(id -> 벡터 맵).

        키워드/그래프로 고른 후보들에 코사인 점수를 매길 때, id 하나씩 get을 호출하는
        N+1 패턴을 피하기 위한 배치 조회다.
        """
        ids = list(ids)
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(ids), _UPSERT_BATCH):
            batch = ids[i : i + _UPSERT_BATCH]
            try:
                got = self._col.get(ids=batch, include=["embeddings"])
            except Exception as e:
                # 한 배치가 실패해도 나머지 배치는 계속 처리한다(부분 결과 허용).
                # 다만 조용히 넘기지 않고 남겨, 일부 점수가 0으로 매겨지는 원인을 추적 가능하게 한다.
                log.warning("get_embeddings batch [%d:%d] failed: %s", i, i + _UPSERT_BATCH, e)
                continue
            embs = got.get("embeddings")
            # 주의: embs는 numpy ndarray일 수 있어 `embs or []`로 쓰면 "배열의 진리값이
            # 모호하다"는 예외가 난다. 그래서 명시적으로 None만 비교해 거른다.
            if embs is None:  # ndarray — avoid `or []` (ambiguous truth value)
                continue
            for gid, emb in zip(got.get("ids", []), embs):
                out[str(gid)] = np.asarray(emb, dtype=np.float32)
        return out

    # 전체 스캔(페이징) 시 한 페이지에 가져올 레코드 수. 큰 인덱스에서도 메모리 사용량을
    # 일정하게 유지하기 위해 offset/limit로 나눠 읽는다.
    _SCAN_PAGE = 5000

    def existing_mtimes(self) -> dict[str, str]:
        """id -> 저장된 폴더 mtime 키(문자열) 맵을 반환. 증분 색인의 "건너뛰기" 판단용.

        mtime을 일부러 "고정 정밀도 문자열"로 저장/비교한다. float mtime은 float64의
        정밀도 경계에 걸려 Chroma 왕복(저장→재조회) 과정에서 미세하게 달라질 수 있는데,
        문자열로 다루면 동등 비교가 정확해진다. 대규모(이미지 단위) 인덱스에서도
        메모리가 폭발하지 않도록 페이징으로 읽는다.
        """
        out: dict[str, str] = {}
        offset = 0
        while True:
            try:
                got = self._col.get(
                    include=["metadatas"], limit=self._SCAN_PAGE, offset=offset
                )
            except Exception as e:
                # 중간에 실패하면 지금까지 모은 결과만 반환(부분 결과라도 증분 판단에 도움).
                # 부분 결과는 일부 레코드가 재색인되게 만들 수 있으므로 경고로 남긴다.
                log.warning("existing_mtimes() scan failed at offset %d (partial result): %s", offset, e)
                return out
            ids = got.get("ids", [])
            for _id, md in zip(ids, got.get("metadatas", []) or []):
                # mtime이 실제로 기록된 레코드만 맵에 담는다(과거 인덱스엔 없을 수 있음).
                if md and md.get("mtime") is not None:
                    out[_id] = str(md["mtime"])
            # 받아온 개수가 페이지 크기보다 작으면 마지막 페이지 → 종료.
            if len(ids) < self._SCAN_PAGE:
                return out
            offset += len(ids)

    def stored_signature(self) -> Optional[tuple[str, int, Optional[str]]]:
        """기존 레코드의 (model_id, embed_dim, granularity) 시그니처를 반환. 불일치 감지용.

        현재 설정의 모델/차원/세분도(granularity)가 인덱스에 저장된 것과 다르면, 그대로
        섞어 쓰면 벡터 공간이 맞지 않으므로 재색인이 필요하다는 것을 이 값으로 판단한다.
        granularity는 그 필드가 기록되기 이전에 만들어진 인덱스에선 None일 수 있다.
        """
        try:
            # 시그니처는 어느 레코드든 동일하므로 한 건만 읽어 비용을 최소화한다.
            got = self._col.get(limit=1, include=["metadatas"])
        except Exception:
            return None
        metas = got.get("metadatas") or []
        # model_id/embed_dim이 둘 다 있는 정상 레코드일 때만 시그니처를 구성.
        if metas and metas[0] and "model_id" in metas[0] and "embed_dim" in metas[0]:
            try:
                gran = metas[0].get("granularity")
                return (
                    str(metas[0]["model_id"]),
                    int(metas[0]["embed_dim"]),
                    # granularity가 없던 옛 인덱스는 None으로 둔다(과거 호환).
                    str(gran) if gran is not None else None,
                )
            except (TypeError, ValueError):
                # 값이 깨져 형 변환에 실패하면 시그니처 없음으로 처리(안전한 기본값).
                return None
        return None

    @staticmethod
    def _hit_from_meta(md: dict, doc: str, score: float) -> FolderHit:
        """저장된 메타데이터/문서/점수를 UI가 쓰는 FolderHit 도메인 객체로 변환.

        각 필드를 방어적으로 캐스팅하고 기본값을 채워, 메타데이터가 일부 누락돼도
        UI가 깨지지 않게 한다.
        """
        md = md or {}
        return FolderHit(
            folder=str(md.get("leaf_folder", "")),
            image_path=str(md.get("representative_image", "")),
            # caption이 비어 있으면 문서(doc) 본문을 대신 사용한다(둘 다 없으면 빈 문자열).
            caption=str(md.get("caption", "") or doc or ""),
            score=score,
            # member_count가 None이거나 0이면 0으로(`or 0`은 None/0 모두 0 처리).
            member_count=int(md.get("member_count", 0) or 0),
            sidecar_text=str(md.get("sidecar_text", "")),
        )

    def query(self, embedding: np.ndarray, k: int) -> list[FolderHit]:
        """질의 임베딩으로 상위 k개의 폴더 히트를 코사인 유사도순으로 반환한다."""
        # 비어 있는 컬렉션에 query하면 오류/빈 형태가 제각각이라, 미리 단락시킨다.
        if self.count() == 0:
            return []
        # 임베딩을 float32 1차원 리스트로 평탄화(Chroma가 요구하는 입력 형식).
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        res = self._col.query(
            query_embeddings=[vec],
            # k는 최소 1 이상으로 보정(0이나 음수가 들어와도 안전).
            n_results=max(1, int(k)),
            include=["metadatas", "documents", "distances"],
        )
        # Chroma는 다중 질의를 가정해 결과를 [[...]]로 한 겹 더 감싼다. 단일 질의이므로
        # [0]으로 첫 질의의 결과만 꺼낸다(None일 때 대비해 [[]] 기본값).
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        hits: list[FolderHit] = []
        for md, doc, dist in zip(metas, docs, dists):
            sim = 1.0 - float(dist)  # 코사인 거리(distance) -> 유사도(similarity) 환산
            # NaN/inf 같은 비정상 값은 0으로, 음수 유사도도 0으로 클램프(점수는 [0,1] 영역).
            score = max(0.0, sim) if math.isfinite(sim) else 0.0
            hits.append(self._hit_from_meta(md, doc, score))
        return hits

    def search_vector(self, embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        """(chroma_id, 코사인 유사도) 리스트 — 하이브리드 검색의 "벡터" 축.

        query()와 달리 FolderHit가 아니라 (id, 점수)만 돌려준다. 레코드가 upsert될 때
        쓰인 것과 "같은 id"를 반환하므로, 같은 id로 색인된 키워드(어휘) 축 결과와
        id 기준으로 융합(fusion)할 수 있다.
        """
        if self.count() == 0:
            return []
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        # 여기서는 id와 distance만 필요하므로 metadatas/documents는 include하지 않아 가볍다.
        res = self._col.query(
            query_embeddings=[vec], n_results=max(1, int(k)), include=["distances"]
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[tuple[str, float]] = []
        for rid, dist in zip(ids, dists):
            sim = 1.0 - float(dist)  # 코사인 거리 -> 유사도
            # query()와 동일한 클램프 규칙으로 점수 정규화.
            out.append((str(rid), max(0.0, sim) if math.isfinite(sim) else 0.0))
        return out

    def all_documents(self) -> list[tuple[str, str]]:
        """모든 레코드의 (chroma_id, 문서) 쌍을 페이징으로 반환 — BM25 인덱스의 입력원.

        어휘 검색(BM25) 인덱스를 외부에서 만들 때 전체 문서 코퍼스를 한 번에 공급한다.
        """
        out: list[tuple[str, str]] = []
        offset = 0
        while True:
            try:
                got = self._col.get(
                    include=["documents"], limit=self._SCAN_PAGE, offset=offset
                )
            except Exception:
                return out
            ids = got.get("ids", [])
            docs = got.get("documents") or []
            for rid, doc in zip(ids, docs):
                # 문서가 None이면 빈 문자열로 정규화(BM25 토크나이저 입력 안정화).
                out.append((str(rid), str(doc or "")))
            if len(ids) < self._SCAN_PAGE:
                return out
            offset += len(ids)

    def fetch(self, ids: Sequence[str]) -> dict[str, FolderHit]:
        """주어진 id들을 FolderHit 객체로 구체화해 (id -> FolderHit) 맵으로 반환.

        벡터 질의를 거치지 않으므로 코사인 점수를 알 수 없어 score는 0.0으로 둔다.
        융합(fused)/그래프로 선정된 id들을 다시 갤러리 행(FolderHit)으로 되살릴 때 쓴다.
        """
        ids = list(ids)
        out: dict[str, FolderHit] = {}
        for i in range(0, len(ids), _UPSERT_BATCH):
            batch = ids[i : i + _UPSERT_BATCH]
            try:
                got = self._col.get(ids=batch, include=["metadatas", "documents"])
            except Exception:
                # 한 배치 실패는 무시하고 나머지를 계속 채운다(부분 결과 허용).
                continue
            gids = got.get("ids", [])
            metas = got.get("metadatas") or []
            docs = got.get("documents") or []
            for gid, md, doc in zip(gids, metas, docs):
                # score=0.0: 점수는 호출부(융합 로직)에서 별도로 매기는 것이 전제.
                out[str(gid)] = self._hit_from_meta(md, doc, 0.0)
        return out
