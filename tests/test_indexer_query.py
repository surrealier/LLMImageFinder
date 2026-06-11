from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index.indexer import Indexer
from imgsearch.store.chroma_store import ChromaStore


def _norm(p: str) -> str:
    return p.replace("\\", "/")


def test_end_to_end_retrieval(sample_ds, tmp_path, n_scenes):
    cfg = AppConfig(dataset_root=str(sample_ds))
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    n = Indexer(emb, cap, store, cfg).build(str(sample_ds), full_rebuild=True)
    assert n == n_scenes
    assert store.count() == n_scenes

    svc = SearchService(emb, store, chat, cfg)

    res = svc.query("밤에 오토바이가 주차되어 있는 이미지", k=3)
    assert res.hits
    assert "motorcycle" in _norm(res.hits[0].folder)
    assert res.summary  # mock summary present

    res2 = svc.query("불과 연기가 있는 이미지", k=3)
    assert "fire" in _norm(res2.hits[0].folder)

    res3 = svc.query("바다 해변 풍경", k=3)
    assert "beach" in _norm(res3.hits[0].folder)


def test_incremental_skip(sample_ds, tmp_path, n_scenes):
    cfg = AppConfig(dataset_root=str(sample_ds))
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    idx = Indexer(emb, cap, store, cfg)
    idx.build(str(sample_ds), full_rebuild=True)
    # second pass: nothing changed -> no new writes, count stable
    written = idx.build(str(sample_ds), full_rebuild=False)
    assert written == 0
    assert store.count() == n_scenes


def test_dim_change_triggers_rebuild(sample_ds, tmp_path, n_scenes):
    # build with mock dim 512
    cfg = AppConfig(dataset_root=str(sample_ds))
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(sample_ds), full_rebuild=True)
    assert store.stored_signature() == ("mock", 512, "folder")

    # switch embed_dim -> an *incremental* build must auto full-rebuild, not crash
    cfg2 = AppConfig(dataset_root=str(sample_ds), embed_dim=256)
    emb2, cap2, chat2 = build_backends(cfg2)
    written = Indexer(emb2, cap2, store, cfg2).build(str(sample_ds), full_rebuild=False)
    assert written == n_scenes
    assert store.stored_signature() == ("mock", 256, "folder")
    assert store.count() == n_scenes


def test_image_granularity_with_yolo_labels(tmp_path):
    from PIL import Image

    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 64), (i * 40, i * 20, 60)).save(ds / f"im{i}.jpg")
        (ds / f"im{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.3\n")  # YOLO labels

    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"),
        index_granularity="image",
        class_names={"0": "사람", "1": "불"},
    )
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    n = Indexer(emb, cap, store, cfg).build(str(tmp_path / "ds"), full_rebuild=True)

    assert n == 4  # one record PER IMAGE, not per folder
    assert store.count() == 4

    svc = SearchService(emb, store, chat, cfg)
    res = svc.query("사람", k=4)
    assert res.hits
    assert any("탐지 객체" in h.caption for h in res.hits)  # caption from labels
    assert all(h.member_count == 1 for h in res.hits)       # per-image records
    # representative_image is the image itself (distinct paths)
    assert len({h.image_path for h in res.hits}) == len(res.hits)


def test_member_images(sample_ds, tmp_path):
    cfg = AppConfig(dataset_root=str(sample_ds))
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(sample_ds), full_rebuild=True)
    svc = SearchService(emb, store, chat, cfg)
    hit = svc.query("불과 연기", k=1).hits[0]
    members = svc.member_images(hit.folder)
    assert len(members) == 3
    assert hit.image_path in members
