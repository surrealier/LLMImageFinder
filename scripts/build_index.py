"""Build the real index for a dataset into the app's persistent store.

Uses the live app-data dirs (so `uv run imgsearch` sees the result), jina-clip-v2 on
GPU, per-image granularity, and YOLO-label captions (class_names from the saved config).
"""

import sys
import time
from pathlib import Path

from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.index.indexer import Indexer
from imgsearch.paths import app_paths, ensure_dirs
from imgsearch.store.chroma_store import ChromaStore

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"E:\Datasets\Argosphoto\학습데이터\v1.0.0"

# Provisional class names from VLM crop-identification (PENDING user confirmation).
# 0/1/2/3/99 are all visually 'phone'; 4 is the rear-camera module; 9 is a lone unclear box.
CLASS_NAMES = {
    "0": "휴대폰",
    "1": "휴대폰",
    "2": "휴대폰",
    "3": "휴대폰",
    "4": "휴대폰 후면 카메라",
    "9": "미상",
    "99": "휴대폰",
}


def main() -> None:
    paths = ensure_dirs(app_paths())
    cfg = AppConfig.load(paths.config_file)
    cfg.dataset_root = ROOT
    # user-set names (config/yaml/GUI) win; provisional VLM names only as fallback
    cfg.class_names = cfg.class_names or CLASS_NAMES
    cfg.index_granularity = "image"
    cfg.embedder_backend = "jina-clip"
    cfg.embedder_model = "jinaai/jina-clip-v2"
    cfg.embedder_device = "auto"
    cfg.embed_dim = 512
    cfg.caption_enabled = True          # label captions; no VLM needed
    cfg.vlm_backend = "mock"
    cfg.chat_backend = "mock"
    cfg.top_k = 48
    cfg.save(paths.config_file)
    print("config saved ->", paths.config_file)
    print("class_names:", cfg.class_names or "(none)")

    embedder, captioner, chat = build_backends(cfg)
    if getattr(embedder, "name", "") != "jina-clip":
        print(f"ERROR: jina-clip embedder unavailable (got {embedder.name}). reason: {cfg.degraded_reason}")
        sys.exit(1)
    print("embedder:", embedder.name, "device:", getattr(embedder, "device", "?"))

    store = ChromaStore(paths.chroma_dir)
    indexer = Indexer(embedder, captioner, store, cfg)

    state = {"t": time.time(), "last": 0}

    def progress(p):
        if p.phase == "index" and (p.current - state["last"] >= 100 or p.current == p.total):
            dt = time.time() - state["t"]
            rate = p.current / dt if dt > 0 else 0
            eta = (p.total - p.current) / rate if rate > 0 else 0
            print(f"  [{p.current}/{p.total}] {rate:.1f} img/s  ETA {eta:.0f}s")
            state["last"] = p.current

    print("indexing…")
    report = indexer.build(ROOT, progress_cb=progress, log_cb=lambda m: print(" ", m), full_rebuild=True)
    print(f"DONE: wrote {report.n_written} records, collection count = {store.count()}")
    if report.skipped:
        print(f"skipped {len(report.skipped)}:")
        for path, err in report.skipped[:20]:
            print("  -", path, "→", err)

    # quick validation queries
    svc = SearchService(embedder, store, chat, cfg)
    for q in ["밤에 오토바이가 주차되어 있는 이미지", "사람이 길에 있는 사진", "스마트폰을 들고 있는 사람", "불과 연기"]:
        res = svc.query(q, k=5)
        print(f"\nQ: {q}")
        for h in res.hits[:5]:
            print(f"   {h.score:0.3f}  {Path(h.image_path).name:42s}  {h.caption[:40]}")


if __name__ == "__main__":
    main()
