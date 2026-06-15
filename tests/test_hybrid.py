import numpy as np
from PIL import Image

from imgsearch.backends.mock import MockCaptioner, MockEmbedder
from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index.indexer import Indexer
from imgsearch.index.lexical import LexicalIndex, lexical_tokens
from imgsearch.store.chroma_store import ChromaStore


# ---------------------------------------------------------------- lexical unit
def test_lexical_tokens_expand_ko_en_and_josa():
    toks = lexical_tokens("오토바이를")
    assert "오토바이" in toks  # josa-stripped
    assert "motorcycle" in toks  # KO->EN expansion


def test_lexical_index_matches_caption_terms():
    ix = LexicalIndex()
    ix.build(
        [
            ("a", "탐지 객체: 오토바이, 사람"),
            ("b", "탐지 객체: 휴대폰"),
            ("c", "보통 밝기의 사진 — 해변, 바다"),
        ]
    )
    assert ix.search("오토바이 사람", 3)[0][0] == "a"
    assert ix.search("휴대폰", 3)[0][0] == "b"
    assert ix.search("없는단어", 3) == []  # no overlap -> no keyword hit
    assert ix.search("", 3) == []


def test_lexical_empty_corpus():
    ix = LexicalIndex()
    ix.build([])
    assert ix.search("사람", 3) == []
    assert ix.size == 0


# ---------------------------------------------------------------- hybrid e2e
def _image_index(tmp_path):
    ds = tmp_path / "ds" / "f"
    ds.mkdir(parents=True)
    # captions will read "탐지 객체: <names>" from these labels
    specs = [("0",), ("0", "1"), ("1",), ("2",)]  # 사람 / 사람+오토바이 / 오토바이 / 휴대폰
    for i, classes in enumerate(specs):
        Image.new("RGB", (48, 48), (i * 50 % 255, i * 30 % 255, 70)).save(ds / f"im{i}.jpg")
        (ds / f"im{i}.txt").write_text("".join(f"{c} 0.5 0.5 0.3 0.3\n" for c in classes))
    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"),
        index_granularity="image",
        class_names={"0": "사람", "1": "오토바이", "2": "휴대폰"},
    )
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(tmp_path / "ds"), full_rebuild=True)
    return cfg, emb, store, chat


def test_keyword_mode_finds_exact_class_term(tmp_path):
    cfg, emb, store, chat = _image_index(tmp_path)
    cfg.search_mode = "keyword"
    svc = SearchService(emb, store, chat, cfg)
    hits = svc.query("휴대폰", k=5).hits
    assert hits, "keyword search should find the phone image"
    assert any("휴대폰" in h.caption for h in hits)
    assert all(h.match == "keyword" for h in hits)
    # score is still a cosine in [0,1] so the threshold filter stays meaningful
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_hybrid_mode_sets_cosine_score_and_provenance(tmp_path):
    cfg, emb, store, chat = _image_index(tmp_path)
    cfg.search_mode = "hybrid"
    svc = SearchService(emb, store, chat, cfg)
    hits = svc.query("오토바이", k=4).hits
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)  # cosine, not RRF
    assert all(h.match in ("vector", "keyword", "both") for h in hits)
    assert any(h.match in ("keyword", "both") for h in hits)  # BM25 contributed
    # ordering is by fused score (desc)
    fused = [h.fused_score for h in hits]
    assert fused == sorted(fused, reverse=True)
    assert hits[0].fused_score > 0.0


def test_vector_mode_unchanged(tmp_path):
    cfg, emb, store, chat = _image_index(tmp_path)
    cfg.search_mode = "vector"
    svc = SearchService(emb, store, chat, cfg)
    hits = svc.query("사람", k=4).hits
    assert hits
    assert all(h.match == "vector" for h in hits)
    assert all(h.fused_score == 0.0 for h in hits)


def test_lexical_index_rebuilds_after_caption_change(tmp_path):
    import os
    import time

    cfg, emb, store, chat = _image_index(tmp_path)
    cfg.search_mode = "keyword"
    svc = SearchService(emb, store, chat, cfg)
    assert svc.query("휴대폰", k=5).hits  # builds + caches lexical index

    # rename class 2 (휴대폰 -> 스마트폰) and refresh captions (documents change,
    # id COUNT is unchanged -> only the revision counter catches this)
    label = next((tmp_path / "ds" / "f").glob("im3.txt"))
    t = time.time() + 50
    os.utime(label, (t, t))
    cfg.class_names = {"0": "사람", "1": "오토바이", "2": "스마트폰"}
    Indexer(MockEmbedder(cfg.embed_dim), MockCaptioner(), store, cfg).refresh_captions(
        str(tmp_path / "ds")
    )
    svc.invalidate_lexical()

    assert svc.query("스마트폰", k=5).hits  # new term now indexed
    assert not svc.query("휴대폰", k=5).hits  # old term gone from captions


def test_lexical_revision_gate_without_explicit_invalidate(tmp_path):
    # even WITHOUT invalidate_lexical(), a store write bumps revision() so the
    # cached index rebuilds on the next query (belt-and-suspenders)
    cfg, emb, store, chat = _image_index(tmp_path)
    cfg.search_mode = "keyword"
    svc = SearchService(emb, store, chat, cfg)
    svc.query("휴대폰", k=5)
    rev_before = store.revision()
    store.delete([i for i in store.all_ids()[:1]])
    assert store.revision() != rev_before
    # next query rebuilds from the new document set (no crash, fewer docs)
    svc.query("사람", k=5)
