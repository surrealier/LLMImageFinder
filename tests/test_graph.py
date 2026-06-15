from PIL import Image

from imgsearch.config import AppConfig, DEFAULT_IMAGE_EXTS
from imgsearch.graph.base import GraphRecord
from imgsearch.graph.builder import build_records
from imgsearch.graph.memory_graph import MemoryGraphStore


def _records():
    return [
        GraphRecord("a.jpg", "/f", ["사람", "오토바이"]),
        GraphRecord("b.jpg", "/f", ["사람", "차"]),
        GraphRecord("c.jpg", "/f", ["사람", "오토바이", "차"]),
        GraphRecord("d.jpg", "/f", ["휴대폰"]),
    ]


def test_memory_graph_counts_and_cooccurrence():
    g = MemoryGraphStore()
    g.build(_records())
    assert g.count() == 4
    assert g.class_counts() == {"사람": 3, "오토바이": 2, "차": 2, "휴대폰": 1}
    co = dict(g.cooccurring("사람"))
    assert co == {"오토바이": 2, "차": 2}
    assert dict(g.cooccurring("휴대폰")) == {}  # appears alone


def test_memory_graph_images_with_all_and_any():
    g = MemoryGraphStore()
    g.build(_records())
    assert g.images_with_all(["사람", "오토바이"]) == ["a.jpg", "c.jpg"]
    assert g.images_with_all(["사람", "오토바이", "차"]) == ["c.jpg"]
    assert g.images_with_all(["사람", "없는클래스"]) == []  # absent class -> empty
    assert g.images_with_all([]) == []
    assert set(g.images_with_any(["오토바이", "휴대폰"])) == {"a.jpg", "c.jpg", "d.jpg"}


def test_memory_graph_edges():
    g = MemoryGraphStore()
    g.build(_records())
    edges = {(a, b): n for a, b, n in g.edges()}
    assert edges[("사람", "오토바이")] == 2
    assert edges[("사람", "차")] == 2
    assert edges[("오토바이", "차")] == 1
    assert ("휴대폰", "사람") not in edges and ("사람", "휴대폰") not in edges


def test_build_records_from_yolo_labels(tmp_path):
    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(ds / "im0.jpg")
    (ds / "im0.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")  # 사람 + 오토바이
    Image.new("RGB", (32, 32), (4, 5, 6)).save(ds / "im1.jpg")
    (ds / "im1.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # 사람
    Image.new("RGB", (32, 32), (7, 8, 9)).save(ds / "bare.jpg")  # no label -> excluded

    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"),
        index_granularity="image",
        class_names={"0": "사람", "1": "오토바이"},
        image_exts=list(DEFAULT_IMAGE_EXTS),
    )
    recs = build_records(str(tmp_path / "ds"), cfg)
    assert len(recs) == 2  # bare.jpg excluded (no boxes)
    g = MemoryGraphStore()
    g.build(recs)
    assert g.class_counts() == {"사람": 2, "오토바이": 1}
    assert dict(g.cooccurring("사람")) == {"오토바이": 1}


def test_build_records_missing_root_is_empty():
    cfg = AppConfig(dataset_root="", index_granularity="image")
    assert build_records("", cfg) == []
    assert build_records("/no/such/path/xyz", cfg) == []


def test_kuzu_matches_memory_backend(tmp_path):
    import pytest

    pytest.importorskip("kuzu")  # only when the [graph] extra is installed
    from imgsearch.graph.kuzu_graph import KuzuGraphStore

    recs = _records()
    mem = MemoryGraphStore()
    mem.build(recs)
    ku = KuzuGraphStore(tmp_path / "g")
    ku.build(recs)

    assert ku.count() == mem.count()
    assert ku.class_counts() == mem.class_counts()
    assert sorted(ku.cooccurring("사람")) == sorted(mem.cooccurring("사람"))
    assert ku.images_with_all(["사람", "오토바이"]) == mem.images_with_all(["사람", "오토바이"])
    assert ku.images_with_all(["사람", "오토바이", "차"]) == mem.images_with_all(["사람", "오토바이", "차"])
    assert set(ku.images_with_any(["오토바이", "휴대폰"])) == set(mem.images_with_any(["오토바이", "휴대폰"]))
    assert {(a, b): n for a, b, n in ku.edges()} == {(a, b): n for a, b, n in mem.edges()}

    # rebuild is idempotent (drops + recreates the on-disk db)
    ku.build(recs)
    assert ku.count() == 4
