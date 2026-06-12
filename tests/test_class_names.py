import numpy as np
import pytest
from PIL import Image

from imgsearch.config import (
    AppConfig,
    DEFAULT_IMAGE_EXTS,
    load_class_names_yaml,
    save_class_names_yaml,
)
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index import labels
from imgsearch.index.indexer import Indexer
from imgsearch.store.chroma_store import ChromaStore


# ---------------------------------------------------------------- yaml io
def test_yaml_roundtrip(tmp_path):
    p = tmp_path / "class_names.yaml"
    names = {"0": "사람", "4": "휴대폰 후면 카메라", "99": "기타"}
    save_class_names_yaml(p, names)
    assert load_class_names_yaml(p) == names
    text = p.read_text(encoding="utf-8")
    assert "사람" in text  # allow_unicode (not escaped)


def test_yaml_ultralytics_list_format(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text("names:\n- person\n- motorcycle\n", encoding="utf-8")
    assert load_class_names_yaml(p) == {"0": "person", "1": "motorcycle"}


def test_yaml_missing_or_invalid(tmp_path):
    assert load_class_names_yaml(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    assert load_class_names_yaml(bad) == {}


# ---------------------------------------------------------------- dataset helpers
def _mini_yolo_ds(tmp_path, n=4):
    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 64), (i * 40, i * 20, 60)).save(ds / f"im{i}.jpg")
        (ds / f"im{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.3\n")
    return tmp_path / "ds"


def test_scan_class_ids(tmp_path):
    root = _mini_yolo_ds(tmp_path)
    counts = labels.scan_class_ids(root, DEFAULT_IMAGE_EXTS)
    assert counts == {0: 2, 1: 2}


# ---------------------------------------------------------------- caption refresh
class _MustNotEmbed:
    """Embedder stub proving refresh_captions never embeds."""

    name = "must-not-embed"
    dim = 512

    def embed_image(self, paths):
        raise AssertionError("refresh_captions must not embed images")

    def embed_text(self, texts):
        raise AssertionError("refresh_captions must not embed text")


def test_refresh_captions_updates_names_without_reembedding(tmp_path):
    root = _mini_yolo_ds(tmp_path)
    cfg = AppConfig(
        dataset_root=str(root), index_granularity="image", class_names={"0": "사람", "1": "차"}
    )
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(root), full_rebuild=True)
    assert store.stored_signature() == ("mock", 512, "image")

    svc = SearchService(emb, store, chat, cfg)
    assert any("사람" in h.caption for h in svc.query("x", k=4).hits)

    before = store._col.get(include=["embeddings"])
    emb_before = {i: np.asarray(e) for i, e in zip(before["ids"], before["embeddings"])}

    # rename classes; refresh with an embedder that MUST NOT be called
    cfg.class_names = {"0": "보행자", "1": "오토바이"}
    n = Indexer(_MustNotEmbed(), None, store, cfg).refresh_captions(str(root))
    assert n == 4
    assert store.count() == 4
    # stored embedding signature preserved (not overwritten by the stub's name)
    assert store.stored_signature() == ("mock", 512, "image")

    # vectors must be bit-identical (no re-embedding, no default-EF corruption)
    after = store._col.get(include=["embeddings"])
    emb_after = {i: np.asarray(e) for i, e in zip(after["ids"], after["embeddings"])}
    for rid, vec in emb_before.items():
        np.testing.assert_allclose(emb_after[rid], vec, rtol=0, atol=1e-7)

    hits = svc.query("x", k=4).hits
    assert any("보행자" in h.caption or "오토바이" in h.caption for h in hits)
    assert not any("사람" in h.caption for h in hits)


def test_refresh_preserves_stored_mtime_so_changed_files_still_reembed(tmp_path):
    import os
    import time

    root = _mini_yolo_ds(tmp_path)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    idx = Indexer(emb, cap, store, cfg)
    idx.build(str(root), full_rebuild=True)
    mtimes_at_build = store.existing_mtimes()

    # an image changes AFTER the build…
    changed = next((root / "folderA").glob("im0.jpg"))
    t = time.time() + 100
    os.utime(changed, (t, t))

    # …then the user renames classes (refresh). Refresh must NOT stamp the new
    # mtime — otherwise the next incremental build would skip the re-embed.
    cfg.class_names = {"0": "보행자"}
    Indexer(_MustNotEmbed(), None, store, cfg).refresh_captions(str(root))
    assert store.existing_mtimes() == mtimes_at_build

    # the next incremental build re-embeds exactly the changed file
    report = idx.build(str(root), full_rebuild=False)
    assert report.n_written == 1


def test_refresh_leaves_unlabeled_records_untouched(tmp_path):
    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(ds / "labeled.jpg")
    (ds / "labeled.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    Image.new("RGB", (64, 64), (90, 10, 70)).save(ds / "bare.jpg")  # no label/sidecar

    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"), index_granularity="image", class_names={"0": "사람"}
    )
    emb, cap, chat = build_backends(cfg)  # mock captioner captions the bare image
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(tmp_path / "ds"), full_rebuild=True)

    def caption_of(name: str) -> str:
        got = store._col.get(include=["metadatas"])
        for md in got["metadatas"]:
            if name in md["representative_image"]:
                return md["caption"]
        raise AssertionError(name)

    bare_before = caption_of("bare.jpg")
    assert bare_before  # captioner produced something at build time

    cfg.class_names = {"0": "보행자"}
    n = Indexer(_MustNotEmbed(), None, store, cfg).refresh_captions(str(tmp_path / "ds"))
    assert n == 1  # only the labeled record was updated
    assert "보행자" in caption_of("labeled.jpg")
    assert caption_of("bare.jpg") == bare_before  # stored caption survived


def test_label_edit_invalidates_incremental_skip(tmp_path):
    import os
    import time

    root = _mini_yolo_ds(tmp_path)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    idx = Indexer(emb, cap, store, cfg)
    idx.build(str(root), full_rebuild=True)
    assert idx.build(str(root), full_rebuild=False).n_written == 0  # clean skip

    # re-annotate one label file (image untouched) -> must re-process that record
    label = next((root / "folderA").glob("im1.txt"))
    label.write_text("0 0.4 0.4 0.2 0.2\n0 0.6 0.6 0.2 0.2\n")
    t = time.time() + 100
    os.utime(label, (t, t))
    assert idx.build(str(root), full_rebuild=False).n_written == 1


def test_refresh_requires_image_granularity(tmp_path):
    root = _mini_yolo_ds(tmp_path)
    cfg = AppConfig(dataset_root=str(root), index_granularity="folder")
    store = ChromaStore(tmp_path / "chroma")
    with pytest.raises(ValueError):
        Indexer(_MustNotEmbed(), None, store, cfg).refresh_captions(str(root))


def test_refresh_on_empty_store(tmp_path):
    root = _mini_yolo_ds(tmp_path)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image")
    store = ChromaStore(tmp_path / "chroma")
    assert Indexer(_MustNotEmbed(), None, store, cfg).refresh_captions(str(root)) == 0
