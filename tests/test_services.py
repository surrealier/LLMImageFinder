import numpy as np
import pytest
from PIL import Image

from imgsearch.backends.mock import MockEmbedder
from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index.indexer import Indexer
from imgsearch.store.chroma_store import ChromaStore


class _FailingChat:
    """A chat backend whose server just died mid-session."""

    def refine_query(self, ko_text: str) -> str:
        raise RuntimeError("connection refused")

    def summarize(self, query, hits) -> str:
        raise RuntimeError("connection refused")


class _WhitespaceChat:
    def refine_query(self, ko_text: str) -> str:
        return "   "

    def summarize(self, query, hits) -> str:
        return ""


class _MustNotChat:
    def refine_query(self, ko_text: str) -> str:
        return ko_text

    def summarize(self, query, hits) -> str:
        raise AssertionError("summarize must not be called when disabled")


class _MustNotEmbed:
    name = "must-not-embed"
    dim = 512

    def embed_image(self, paths):
        raise AssertionError("stored embedding should have been reused")

    def embed_text(self, texts):
        raise AssertionError("text embedding must not run here")


def _build_image_index(tmp_path, n=5):
    ds = tmp_path / "ds" / "folderA"
    ds.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 64), (i * 37 % 255, i * 19 % 255, 60)).save(ds / f"im{i}.jpg")
        (ds / f"im{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.3\n")
    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"),
        index_granularity="image",
        class_names={"0": "사람", "1": "차"},
    )
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(tmp_path / "ds"), full_rebuild=True)
    return cfg, emb, store, chat


def test_query_survives_dead_chat_backend(tmp_path):
    cfg, emb, store, _ = _build_image_index(tmp_path)
    svc = SearchService(emb, store, _FailingChat(), cfg)
    res = svc.query("사람", k=4)
    assert res.hits  # search still works
    assert res.refined == "사람"  # raw text used when refine fails
    assert res.summary == ""  # summary silently dropped


def test_whitespace_refinement_keeps_original(tmp_path):
    cfg, emb, store, _ = _build_image_index(tmp_path)
    svc = SearchService(emb, store, _WhitespaceChat(), cfg)
    res = svc.query("사람", k=2)
    assert res.refined == "사람"


def test_empty_query_never_embeds(tmp_path):
    cfg = AppConfig(index_granularity="image")
    store = ChromaStore(tmp_path / "chroma")
    svc = SearchService(_MustNotEmbed(), store, None, cfg)
    res = svc.query("   ")
    assert res.hits == []
    assert "검색어" in res.summary


def test_summarize_disabled_never_called(tmp_path):
    cfg, emb, store, _ = _build_image_index(tmp_path)
    cfg.summarize_enabled = False
    svc = SearchService(emb, store, _MustNotChat(), cfg)
    res = svc.query("사람", k=2)
    assert res.hits
    assert res.summary == ""


def test_query_by_example_reuses_stored_embedding(tmp_path):
    cfg, emb, store, chat = _build_image_index(tmp_path, n=5)
    svc = SearchService(emb, store, chat, cfg)
    example = svc.query("사람", k=5).hits[0]

    # an embedder that refuses to embed proves the stored vector is reused
    svc2 = SearchService(_MustNotEmbed(), store, None, cfg)
    res = svc2.query_by_example(example, k=3)
    assert res.hits
    assert len(res.hits) <= 3
    assert all(h.image_path != example.image_path for h in res.hits)  # self excluded
    assert "유사 이미지" in res.query


def test_query_by_example_folder_granularity(tmp_path, sample_ds):
    cfg = AppConfig(dataset_root=str(sample_ds))
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(sample_ds), full_rebuild=True)
    svc = SearchService(emb, store, chat, cfg)
    example = svc.query("불과 연기", k=1).hits[0]
    res = SearchService(_MustNotEmbed(), store, None, cfg).query_by_example(example, k=4)
    assert res.hits
    assert all(h.folder != example.folder for h in res.hits) or all(
        h.image_path != example.image_path for h in res.hits
    )