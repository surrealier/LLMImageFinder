# Changelog

## v0.3.1 — 2026-06-22

Release-hardening pass for a polished public release: branding, first-run UX,
packaging metadata, observability, CI, and English documentation. No behavior
changes to search/indexing — the v0.2.0-audited core is unchanged.

### Branding & UX
- **App icon** — a code-generated magnifier icon (no shipped binary asset) now
  appears in the title bar and the Windows taskbar (explicit AppUserModelID).
- **Help menu** — `Help → About` (version, description, repo/docs links, MIT) and
  `Help → Keyboard shortcuts` (a single list of every binding).
- **Version is visible** — shown in the status bar and the About dialog.
- **First-run empty-state** — the gallery now shows guidance when there are no
  results: "no index yet → Generate sample dataset / Build index", "ready → describe
  an image", or, after a search, "no matches → lower Score ≥ / rephrase".
- **Clearer copy** — plainer tooltips for the search-mode and agent toggles, a
  reassurance that the dataset is never modified, and an actionable "no results"
  message.

### Packaging & release hygiene
- **Single source of truth for the version** — `pyproject.toml` reads the version
  from `imgsearch/__init__.py` (`[tool.hatch.version]`); the stale `0.1.0` dunder is
  fixed. A test asserts the installed metadata matches `__version__`.
- **Full project metadata** — `authors`, SPDX `license`, `keywords`, trove
  `classifiers`, and `project.urls` (homepage/repository/issues/changelog).
- **Continuous integration** — GitHub Actions runs the test suite + the offscreen GUI
  smoke on Windows and Linux (Python 3.12) on every push/PR.

### Observability
- `ChromaStore` no longer swallows exceptions silently — `count`, `all_ids`,
  `get_embedding`, `get_embeddings` (per batch), and `existing_mtimes` now log a
  warning while keeping their safe fallbacks, so a corrupt/locked store is diagnosable
  instead of silently looking "empty".

### Docs
- `docs/ARCHITECTURE.md` and this changelog translated to English (the UI and outputs
  were already English; source-code comments remain Korean by design).

### Tests
- 84 → 90: package-metadata/version consistency, the About and Keyboard-shortcuts
  dialogs, the gallery empty-state, and app-icon construction.

## v0.3.0 — 2026-06-15

RAG / VectorDB / multi-agent / hybrid search / GraphDB / vLLM in a single release.
Designed by five architects with adversarial verification of cross-integration risks,
and validated on a live 1,147-image jina-clip index across all four features.

### Hybrid search (vector + keyword RRF)
- Adds **BM25 keyword search** (`rank-bm25`, pure Python) on top of the existing CLIP
  cosine search, fused via **Reciprocal Rank Fusion** (k=60). Switch instantly between
  Hybrid / Vector / Keyword in the gallery header's **search mode** combo.
- `FolderHit.score` stays cosine in every mode (preserving the score-threshold filter's
  meaning); ordering uses `fused_score`, and the match source is shown in `match`
  (vector / keyword / both).
- BM25 matches class names in captions exactly ("the rear camera of a phone", etc.) —
  candidates are selected by token overlap even for frequent terms (avoiding the
  Okapi-IDF-becomes-0/negative problem that dropped common terms).
- The Korean tokenizer reuses koutil (particle splitting + KO→EN synonyms). The keyword
  index is gated on the store's write-revision — it rebuilds automatically after caption
  updates/deletes.

### Embedded GraphDB (object co-occurrence graph)
- Builds an **image–object–folder** graph from the dataset's YOLO labels. Two backends
  behind one protocol: **memory** (pure Python, the dependency-free default) and **kuzu**
  (embedded GraphDB, Cypher, the `[graph]` extra). Falls back to memory automatically if
  kuzu is unavailable.
- Toolbar **Object Graph** dialog: image count per class, objects that **co-occur** with a
  selected object, and "show images containing all selected objects" (AND) → results in
  the gallery.
- Graph↔index joins always go through `image_id()` (independent of path case/separator),
  and the filter is safely skipped on a root mismatch.

### In-app multi-agent search (A2A pipeline)
- Agentic search that runs **plan → hybrid search → graph filter → summarize**, with each
  step traced in the chat (`🧭 Plan / 🔍 Hybrid search / 🕸 Graph filter / ✍ Summary`). Enabled
  with the header **Agent** toggle.
- The planner decomposes a query into semantic text + required/excluded objects. It uses
  vLLM's `plan()` (JSON) and falls back deterministically to the koutil rule-based planner
  — fully functional even on the mock backend.
- If the graph is unbuilt/mismatched, the filter is skipped and hybrid results are kept
  (no empty results).

### vLLM real mode
- Adds `OpenAICompatChatLLM.plan()` (robust JSON parsing, code-fence stripping, fallback).
- `scripts/check_vllm.py` — checks the connect / refine / summarize / plan / (optional)
  caption round-trips as PASS/FAIL.

### Infrastructure / threading
- The object graph builds in the background via `GraphBuildWorker` (rebuilt automatically
  after indexing). The agent trace goes through the worker's `trace` signal → a queued
  connection to the GUI thread (no cross-thread widget access). On shutdown, graph workers
  are drained and the kuzu lock is released.
- Adds an **object-graph backend** (memory/kuzu) choice in Settings.

### Tests
- 60 → 84: hybrid (per-mode score/source/RRF/revision rebuild), graph (memory/kuzu
  parity, co-occurrence, AND, builder), agent (planner, graph filter, exclusion, mismatch
  fallback, callback threading, plan fallback).

## v0.2.0 — 2026-06-12

A release finalized through QC/QA, UI/UX, and performance audits (multi-agent review +
cross-verification).

### Performance
- **Batched embedding**: per-image indexing now embeds images in batches of 32 in one GPU
  call (was: 1,147 single-image calls). Failed rows are retried individually, then recorded
  in the skip list.
- **Background model loading**: the jina-clip weight load (10–40 s) moved to a worker thread
  — the window opens instantly with a "Loading model…" chip in the status bar, and search
  becomes available once it completes.
- **Async thumbnail prewarming**: JPEG decode/save moved off the index thread into a separate
  thread pool (so indexing focuses on embedding).
- **Settings-save optimization**: the model is not reloaded when backend-related settings are
  unchanged (fixing a multi-second stall on top_k / thumbnail-size changes).
- **Gallery paint optimization**: thumbnails are scaled once on delivery — hover/scroll
  repaints are a plain blit.
- **Paged store scans**: `existing_mtimes()` pages in chunks of 5,000 (bounding memory on
  large indexes) and removes one redundant full scan on the caption-refresh path.

### New features
- **Query history**: recall previous queries with ↑/↓ in the input box (stored as JSON in the
  app data, up to 200).
- **Tile context menu**: view image · find similar images · copy path/folder path/caption ·
  reveal in Explorer.
- **Find similar images**: query-by-example reusing the stored embedding — instant, with no
  model call.
- **Viewer YOLO box overlay**: show bounding boxes + class names with the B key or the
  "Labels" button (per-class color, zoom-independent thickness).
- **Viewer ranked navigation**: on an image-granularity index, ←/→ move in search-ranking
  order, with `Result 12/48 · score 0.43 · caption` shown at the bottom.
- **Result count (k) / score-threshold controls**: adjust instantly in the gallery header —
  the threshold filters without re-searching.
- **Result export**: save CSV (utf-8-sig, Excel-friendly) / copy the path list to the
  clipboard.
- **Auto-prune deleted files**: incremental indexing removes records for files that vanished
  from disk (preventing dead thumbnails / broken "open folder").
- **Indexing error report**: the skipped-file list is shown under "details" on completion.
- **Index info dialog**: record count · model/dimension/granularity · last build time/duration ·
  the dataset path at build time (warns + offers rebuild on mismatch) · storage usage.
- **Shortcuts**: F5 update index, Ctrl+L focus the search box, Enter open the viewer,
  Ctrl+C copy path, Ctrl+E reveal in Explorer.

### Bug fixes
- Viewer wheel-zoom turned into scrolling after one zoom (now always zooms, via a viewport
  event filter).
- The "open folder" button stayed enabled when results were cleared.
- Added guidance for clicking "Generate sample dataset" during indexing and for duplicate
  build requests that were silently ignored.
- Unreadable images stayed on "Loading image…" forever → now show "thumbnail unavailable".
- The app could fail to ever exit if a thumbnail decode was wedged on shutdown.
- Settings: API-key masking, an embed_dim cap of 1024 (jina-clip compatible), tooltips on key
  fields.
- The icon-only toolbar made build/rebuild hard to tell apart (text labels + a warning color
  on rebuild).
- Thumbnail-cache write races → prevented with a temp file + atomic replace.
- Fixed a "complete" message shown when indexing was cancelled.

### Tests
- 41 → 60: config persistence, chat-backend fault tolerance, query-by-example (proving stored-
  embedding reuse), batch-embedding call count/vector identity, deleted-record pruning, score
  clamping, store paging, update_meta with missing ids, and more.

## v0.1.0 — 2026-06-10

Initial release: PySide6 GUI, ChromaDB index (folder/image granularity), jina-clip-v2 Korean
text→image search, YOLO-label captions + user class names (GUI/CLI/yaml), a vLLM adapter, and
a mock-first design.
