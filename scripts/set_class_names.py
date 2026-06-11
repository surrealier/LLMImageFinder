"""Set YOLO class names and apply them to the existing index WITHOUT re-embedding.

Usage:
  uv run python scripts/set_class_names.py 0=사람 1=오토바이 99=기타
  uv run python scripts/set_class_names.py --show
  uv run python scripts/set_class_names.py 0=사람 --no-refresh

Names are saved to the app-data config.json + class_names.yaml (never into the
dataset folder). With an existing image-granularity index, captions are refreshed
in seconds (metadata update only).
"""

import sys

from imgsearch.backends.mock import MockCaptioner, MockEmbedder
from imgsearch.config import AppConfig, save_class_names_yaml
from imgsearch.index.indexer import Indexer
from imgsearch.paths import app_paths, ensure_dirs
from imgsearch.store.chroma_store import ChromaStore


def main() -> int:
    paths = ensure_dirs(app_paths())
    cfg = AppConfig.load(paths.config_file)

    args = [a for a in sys.argv[1:]]
    refresh = "--no-refresh" not in args
    show_only = "--show" in args
    pairs = [a for a in args if "=" in a]

    if show_only or not pairs:
        print("dataset_root:", cfg.dataset_root or "(미설정)")
        print("granularity :", cfg.index_granularity)
        print("class_names :")
        for k in sorted(cfg.class_names, key=lambda s: (len(s), s)):
            print(f"  {k} = {cfg.class_names[k]}")
        if not pairs:
            print("\n사용법: uv run python scripts/set_class_names.py 0=사람 1=오토바이 …")
        return 0

    for pair in pairs:
        cid, _, name = pair.partition("=")
        cid, name = cid.strip(), name.strip()
        if not cid or not name:
            print(f"무시됨 (형식 오류): {pair!r}")
            continue
        if cid.isdigit():
            cid = str(int(cid))  # canonicalize ('007' -> '7') to match label ids
        cfg.class_names[cid] = name
        print(f"  {cid} = {name}")

    cfg.save(paths.config_file)
    save_class_names_yaml(paths.class_names_file, cfg.class_names)
    print("저장됨:", paths.config_file)
    print("저장됨:", paths.class_names_file)

    if refresh and cfg.dataset_root and cfg.index_granularity == "image":
        store = ChromaStore(paths.chroma_dir)
        if store.count() > 0:
            print(f"\n캡션 갱신 중… ({store.count()}개 레코드, 재임베딩 없음)")
            # refresh never embeds, so a lightweight mock embedder suffices here
            indexer = Indexer(MockEmbedder(cfg.embed_dim), MockCaptioner(), store, cfg)
            n = indexer.refresh_captions(cfg.dataset_root, log_cb=lambda m: print(" ", m))
            print(f"완료: {n}개 캡션 갱신")
            got = store._col.get(limit=3, include=["metadatas"])
            for md in got.get("metadatas") or []:
                print("  예시:", (md or {}).get("caption", "")[:70])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
