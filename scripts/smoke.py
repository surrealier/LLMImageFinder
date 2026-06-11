"""Headless end-to-end smoke test of the mock pipeline (no GUI, no ML)."""

import tempfile
from pathlib import Path

from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index.indexer import Indexer
from imgsearch.sample_data import generate_sample_dataset
from imgsearch.store.chroma_store import ChromaStore


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="imgsearch_smoke_"))
    ds = generate_sample_dataset(tmp / "data", per_scene=4)
    print("dataset:", ds)

    cfg = AppConfig(dataset_root=str(ds))
    embedder, captioner, chat = build_backends(cfg)
    store = ChromaStore(tmp / "chroma")

    idx = Indexer(embedder, captioner, store, cfg)
    n = idx.build(str(ds), log_cb=lambda m: print("  log:", m), full_rebuild=True)
    print(f"indexed records written={n} collection_count={store.count()}")

    svc = SearchService(embedder, store, chat, cfg)
    for q in [
        "밤에 오토바이가 주차되어 있는 이미지",
        "사람이 대로변에 돌아다니는 이미지",
        "불과 연기가 있는 이미지",
        "바다 해변 풍경",
    ]:
        res = svc.query(q, k=3)
        print(f"\nQ: {q}\n   refined: {res.refined}")
        for h in res.hits:
            print(f"   {h.score:0.3f}  {Path(h.folder).name:30s}  members={h.member_count}  {h.caption[:42]}")
        print("   summary:", res.summary[:90])

    # member navigation
    if svc.count():
        top = svc.query("불과 연기", k=1).hits[0]
        members = svc.member_images(top.folder)
        print(f"\nmembers of {Path(top.folder).name}: {len(members)} images")
    print("\nOK")


if __name__ == "__main__":
    main()
