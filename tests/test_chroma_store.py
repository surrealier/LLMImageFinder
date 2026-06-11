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
