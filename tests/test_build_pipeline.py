"""Batched embedding, vanished-file pruning, and skip reporting in Indexer.build."""

import numpy as np
from PIL import Image

from imgsearch.backends.mock import MockCaptioner, MockEmbedder
from imgsearch.config import AppConfig
from imgsearch.index.indexer import Indexer, image_id
from imgsearch.store.chroma_store import ChromaStore


class _CountingEmbedder(MockEmbedder):
    """Records the size of every embed_image call."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__(dim)
        self.calls: list[int] = []

    def embed_image(self, paths):
        self.calls.append(len(paths))
        return super().embed_image(paths)


class _PoisonRowEmbedder(_CountingEmbedder):
    """Returns NaN for one path in batch mode AND on the single retry,
    simulating an image the model cannot embed."""

    def __init__(self, poison_path: str, dim: int = 512) -> None:
        super().__init__(dim)
        self._poison = poison_path

    def embed_image(self, paths):
        vecs = super().embed_image(paths)
        for i, p in enumerate(paths):
            if p == self._poison:
                vecs[i] = np.nan
        return vecs


def _mk_ds(tmp_path, n):
    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    paths = []
    for i in range(n):
        p = ds / f"im{i:03d}.jpg"
        Image.new("RGB", (48, 48), (i * 7 % 255, i * 13 % 255, 40)).save(p)
        (ds / f"im{i:03d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        paths.append(p)
    return tmp_path / "ds", paths


def test_image_build_batches_embedding_calls(tmp_path):
    root, _ = _mk_ds(tmp_path, 70)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb = _CountingEmbedder()
    store = ChromaStore(tmp_path / "chroma")
    report = Indexer(emb, MockCaptioner(), store, cfg).build(str(root), full_rebuild=True)

    assert report.n_written == 70
    assert store.count() == 70
    # 70 images at batch size 32 -> 3 calls (32+32+6), never 70 single calls
    assert len(emb.calls) == 3, emb.calls
    assert max(emb.calls) <= 32 and sum(emb.calls) == 70


def test_batched_vectors_match_single_image_embedding(tmp_path):
    root, paths = _mk_ds(tmp_path, 40)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb = MockEmbedder()
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, MockCaptioner(), store, cfg).build(str(root), full_rebuild=True)

    # stored vectors must equal what a per-image call produces (no batch skew)
    for p in (paths[0], paths[17], paths[39]):
        stored = store.get_embedding(image_id(str(p)))
        single = emb.embed_image([str(p)])[0]
        np.testing.assert_allclose(stored, single, rtol=0, atol=1e-6)


def test_unembeddable_image_is_skipped_and_reported(tmp_path):
    root, paths = _mk_ds(tmp_path, 10)
    poison = str(paths[4])
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb = _PoisonRowEmbedder(poison)
    store = ChromaStore(tmp_path / "chroma")
    report = Indexer(emb, MockCaptioner(), store, cfg).build(str(root), full_rebuild=True)

    assert report.n_written == 9
    assert store.count() == 9
    assert len(report.skipped) == 1
    assert report.skipped[0][0] == poison
    assert store.get_embedding(image_id(poison)) is None


def test_incremental_build_prunes_deleted_files(tmp_path):
    root, paths = _mk_ds(tmp_path, 6)
    cfg = AppConfig(dataset_root=str(root), index_granularity="image", class_names={"0": "사람"})
    emb = MockEmbedder()
    store = ChromaStore(tmp_path / "chroma")
    idx = Indexer(emb, MockCaptioner(), store, cfg)
    idx.build(str(root), full_rebuild=True)
    assert store.count() == 6

    # two files vanish from disk (deleted/moved between sessions)
    paths[1].unlink()
    paths[1].with_suffix(".txt").unlink()
    paths[3].unlink()
    paths[3].with_suffix(".txt").unlink()

    report = idx.build(str(root), full_rebuild=False)
    assert report.pruned == 2
    assert report.n_written == 0  # nothing changed otherwise
    assert store.count() == 4
    assert store.get_embedding(image_id(str(paths[1]))) is None


def test_folder_granularity_prunes_deleted_folders(tmp_path, sample_ds):
    import shutil

    # private copy — sample_ds is session-scoped and must never be mutated
    ds = tmp_path / "ds_copy"
    shutil.copytree(sample_ds, ds)

    cfg = AppConfig(dataset_root=str(ds))
    emb = MockEmbedder()
    store = ChromaStore(tmp_path / "chroma")
    idx = Indexer(emb, MockCaptioner(), store, cfg)
    first = idx.build(str(ds), full_rebuild=True)
    n0 = store.count()
    assert first.n_written == n0

    # remove one whole leaf folder
    victim = next(d for d in ds.rglob("*") if d.is_dir() and any(d.glob("*.jpg")))
    shutil.rmtree(victim)

    report = idx.build(str(ds), full_rebuild=False)
    assert report.pruned >= 1
    assert store.count() == n0 - report.pruned
