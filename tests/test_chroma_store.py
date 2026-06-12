import numpy as np

from imgsearch.store.chroma_store import ChromaStore, folder_id


def _meta(folder: str, mtime: float = 1.0) -> dict:
    return {
        "leaf_folder": folder,
        "representative_image": folder + "/x.jpg",
        "member_count": 3,
        "caption": "설명 " + folder,
        "sidecar_text": "",
        "mtime": mtime,
    }


def test_upsert_query_roundtrip(tmp_path):
    store = ChromaStore(tmp_path / "c")
    ids = [folder_id("/a/f1"), folder_id("/a/f2")]
    embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    store.upsert(ids, embs, ["d1", "d2"], [_meta("/a/f1"), _meta("/a/f2")])
    assert store.count() == 2

    hits = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), 2)
    assert hits[0].folder == "/a/f1"
    assert hits[0].score > hits[1].score
    assert 0.0 <= hits[0].score <= 1.0


def test_idempotent_upsert(tmp_path):
    store = ChromaStore(tmp_path / "c")
    ids = [folder_id("/a/f1")]
    embs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    store.upsert(ids, embs, ["d"], [_meta("/a/f1")])
    store.upsert(ids, embs, ["d"], [_meta("/a/f1")])
    assert store.count() == 1


def test_existing_mtimes_and_clear(tmp_path):
    store = ChromaStore(tmp_path / "c")
    fid = folder_id("/a/f1")
    store.upsert([fid], np.array([[1.0, 0.0]], dtype=np.float32), ["d"], [_meta("/a/f1", "42.000")])
    assert store.existing_mtimes().get(fid) == "42.000"
    store.clear()
    assert store.count() == 0


def test_query_empty(tmp_path):
    store = ChromaStore(tmp_path / "c")
    assert store.query(np.array([1.0, 0.0], dtype=np.float32), 5) == []


def test_score_clamped_and_k_larger_than_count(tmp_path):
    store = ChromaStore(tmp_path / "c")
    ids = [folder_id("/a/f1"), folder_id("/a/f2")]
    embs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    store.upsert(ids, embs, ["d1", "d2"], [_meta("/a/f1"), _meta("/a/f2")])

    # k far beyond the collection size must return everything, not raise
    hits = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), 50)
    assert len(hits) == 2
    # the opposite-direction vector has cosine similarity -1 -> clamped to 0
    assert hits[1].score == 0.0
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_delete_and_get_embedding(tmp_path):
    store = ChromaStore(tmp_path / "c")
    ids = [folder_id("/a/f1"), folder_id("/a/f2")]
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.upsert(ids, embs, ["d1", "d2"], [_meta("/a/f1"), _meta("/a/f2")])

    vec = store.get_embedding(ids[0])
    np.testing.assert_allclose(vec, [1.0, 0.0], atol=1e-6)
    assert store.get_embedding(folder_id("/ghost")) is None

    store.delete([ids[0], folder_id("/ghost")])  # unknown ids must be harmless
    assert store.count() == 1
    assert store.get_embedding(ids[0]) is None
    store.delete([])  # no-op


def test_update_meta_with_missing_ids(tmp_path):
    store = ChromaStore(tmp_path / "c")
    fid = folder_id("/a/f1")
    store.upsert([fid], np.array([[1.0, 0.0]], dtype=np.float32), ["old"], [_meta("/a/f1")])

    ghost = folder_id("/ghost")
    # ids missing from the store are silently dropped — no crash, no new record
    store.update_meta([fid, ghost], ["new", "ghost-doc"], [_meta("/a/f1"), _meta("/ghost")])
    assert store.count() == 1
    got = store._col.get(ids=[fid], include=["documents"])
    assert got["documents"] == ["new"]

    store.update_meta([ghost], ["ghost-doc"], [_meta("/ghost")])  # all missing -> no-op
    assert store.count() == 1


def test_existing_mtimes_paged(tmp_path):
    store = ChromaStore(tmp_path / "c")
    store._SCAN_PAGE = 7  # force several pages
    n = 23
    ids = [folder_id(f"/p/f{i}") for i in range(n)]
    embs = np.zeros((n, 4), dtype=np.float32)
    embs[:, 0] = 1.0
    store.upsert(ids, embs, ["d"] * n, [_meta(f"/p/f{i}", f"{i}.000") for i in range(n)])
    got = store.existing_mtimes()
    assert len(got) == n
    assert got[ids[5]] == "5.000"
