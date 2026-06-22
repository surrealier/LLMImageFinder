# Architecture and Technical Philosophy

> `imgsearch` — a local PySide6 desktop app for searching image datasets with natural-language queries (Korean or English).
> This document explains how the app's core technologies (**RAG · VectorDB · hybrid search · GraphDB · multi-agent · vLLM**)
> are *composed* in the code and *why* they are designed the way they are.

---

## 0. At a Glance

| Technology | One-line definition | Where it lives (key files) |
|---|---|---|
| **VectorDB** | Stores and searches image embeddings in cosine space | `store/chroma_store.py` (ChromaDB) |
| **Embedding** | Maps images and text into the same 512-dimensional space | `backends/jina_clip.py` (real), `backends/mock.py` (mock) |
| **Hybrid search** | Fuses vector (semantic) + keyword (BM25) via RRF | `index/lexical.py` + `core/services.py` |
| **GraphDB** | Object co-occurrence graph (image–object–folder) | `graph/*` (memory/kuzu) |
| **RAG** | Korean summary grounded in search results | `core/services.py` + `backends/openai_compat.py` |
| **Multi-agent** | Plan → search → graph filter → summarize A2A pipeline | `core/agentic.py` |
| **vLLM** | LLM performs captioning, summarization, query refinement, planning | `backends/openai_compat.py`, `backends/prompts.py` |

---

## 1. Five Cross-Cutting Principles

Every technology choice follows the five principles below. When reading the code, the answer to "why was it done this way?" is usually found here.

### (1) Mock-first — every feature works instantly, even with zero ML dependencies
- With just `uv sync` (no torch/transformers/kuzu), **indexing, hybrid search, graph, and even the agents all** run.
- The real models are activated only via *additional dependencies* (`[embed]`, `[graph]`) and *configuration*.
- The mock backend is **deterministic** (`backends/mock.py`, the md5 hash bucket in `koutil.py`). This is why tests are reproducible.
- Rationale: validate the entire UX without a GPU or server for demos/offline/CI, and slot in the real backends behind the same interface.

### (2) Adapter/protocol boundary — the app never knows the model libraries directly
- `backends/base.py` defines only the `Embedder`/`Captioner`/`ChatLLM` **Protocol**. The rest of the code sees only these abstractions.
- A backend is selected by string (`"mock"`/`"jina-clip"`/`"vllm"`), and `core/registry.py` constructs the real backend via *lazy import + try/except*,
  but if it fails it **falls back** to the mock and shows the reason in a banner (`cfg.mark_degraded`).
- The graph works the same way: two implementations, memory and kuzu, behind the `GraphStore` protocol in `graph/base.py`.

### (3) Dataset inviolability — the app never writes to the user's dataset folder
- The index, thumbnails, settings, logs, class names, and graph are all stored only in the **app data folder** (`%LOCALAPPDATA%\MarkAny\ImgSearch\`).
- Paths are managed in one place by `paths.py` (`AppPaths`). Tests are isolated via the `IMGSEARCH_HOME` environment variable.
- Even class names are not placed in the dataset but in `class_names.yaml` under the app data (based on user input).

### (4) Non-blocking UI — the GUI thread only draws
- Heavy work (embedding, indexing, model loading, graph building, search) all runs in **QThread workers** (`workers/*`).
- The `ThreadRunner` (`workers/qworker.py`) holds a reference to each worker to prevent GC, and communicates with the GUI only via signals.
- Pitfall note: a worker thread is blocked inside `run()`, so its event loop is starved. That is why the **cancel flag is connected via
  `Qt.DirectConnection`** (a GIL-atomic bool), while the **agent trace is connected via a queued connection** (it accesses widgets, so it runs on the GUI thread).

### (5) Deterministic and verifiable — 90 tests + offscreen UI smoke
- All core behavior is tested headlessly (`tests/`). The UI is driven offscreen by `scripts/ui_smoke.py`.
- The real-model paths are verified live by `scripts/build_index.py` / `check_vllm.py`.

---

## 2. Layer Map

```
                         ┌──────────────────────────────────────────────┐
  user ◀─ chat/gallery ─▶  │  ui/  (main_window · chat_widget · gallery ·  │
                         │       image_viewer · *_dialog · delegate)     │
                         └───────────────┬───────────────────────────────┘
                          signals/slots  │  (heavy work delegated to workers)
                         ┌───────────────▼───────────────────────────────┐
                         │  workers/  (QThread: index · query · graph ·   │
                         │            preload · thumb)                    │
                         └───────────────┬───────────────────────────────┘
                                         │
        ┌────────────────────────────────▼────────────────────────────────┐
        │  core/   services(SearchService: hybrid·RAG) · agentic(A2A)      │
        │          registry(backend factory) · models(dataclass)           │
        └───┬───────────────┬───────────────┬───────────────┬──────────────┘
            │               │               │               │
   ┌────────▼──────┐ ┌──────▼───────┐ ┌─────▼──────┐ ┌───────▼─────────┐
   │ backends/     │ │ index/       │ │ store/     │ │ graph/          │
   │ base(Protocol)│ │ walker·      │ │ chroma_    │ │ base(Protocol)· │
   │ mock·jina_clip│ │ pairing·     │ │ store      │ │ memory·kuzu·    │
   │ openai_compat │ │ repr_select· │ │ (VectorDB) │ │ builder·registry│
   │ prompts       │ │ labels·      │ │            │ │ (GraphDB)       │
   │               │ │ indexer·     │ │            │ │                 │
   │               │ │ lexical(BM25)│ │            │ │                 │
   └───────────────┘ └──────────────┘ └────────────┘ └─────────────────┘
            │
   ┌────────▼───────────────────────────────────────────────────────────┐
   │ cross-cutting: config · paths(app-data) · koutil(Korean/mock) ·      │
   │       thumbs(thumbnail cache) · logging_setup                        │
   └──────────────────────────────────────────────────────────────────────┘
```

Rule: **an upper layer knows only the layers below it.** `core` does not import Qt (which is why headless testing is possible).
`backends`/`graph`/`store` do not know each other (they are connected only through protocols).

---

## 3. Technology-by-Technology Breakdown

### 3.1 VectorDB — ChromaDB (`store/chroma_store.py`)

**Composition.** A single `representatives` collection in cosine space (`hnsw:space=cosine`). One record =
(id, embedding, document=caption, metadata). The id is the sha1 hash of the path, so re-indexing is an **idempotent upsert**.

**Philosophy.**
- *Use only our own embeddings*: never attach Chroma's default embedding function (EF). We insert CLIP/mock vectors directly.
  - Pitfall (this was a real bug): if you do not pass `embeddings=` to `update()`, Chroma **re-embeds** the document
    with the default ONNX MiniLM (384-dim), polluting the space. → `update_meta` reads the stored embedding and puts it back in.
- *A revision counter for change detection*: `_rev` is incremented on every write (upsert/update_meta/delete/clear).
  An `update_meta` that only changes the caption keeps the record count unchanged, so `count()` cannot catch it → the hybrid BM25 cache
  judges staleness via `revision()` (= `(count, _rev)`) and rebuilds accordingly.
- *Incremental indexing*: `existing_mtimes()` reads `id→mtime key` page by page and skips records that have not changed.

**Key methods.** `upsert`, `update_meta` (prevents re-embedding), `query` (cosine top-k → `FolderHit`),
`search_vector` (for hybrid, **returns ids**), `all_documents` (BM25 corpus), `fetch` (id→FolderHit),
`get_embedding`/`get_embeddings` (reuse stored vectors), `revision`.

### 3.2 Embedding — jina-clip-v2 / mock (`backends/jina_clip.py`, `backends/mock.py`)

**Composition.** `jinaai/jina-clip-v2` (multilingual CLIP) runs in-process via `sentence-transformers`, with a 512-dim Matryoshka truncation.
Korean text→image search works directly **without translation**. The mock is a deterministic hash + a bit of real visual signal (downscaled gray/histogram).

**Philosophy.** Text and images must enter *the same space*, L2-normalized, for cosine to be meaningful. Because they are normalized,
**inner product = cosine**, which is why even keyword-only hits compute their score cheaply as `stored vector · query vector` (`services._retrieve`).

### 3.3 Hybrid Search — BM25 + vector + RRF (`index/lexical.py`, `core/services.py`)

**Why.** Vector search captures *semantics* well but can miss *exact terms* like "the rear camera of a phone," and the reverse is also true.
Combining the two signals lets each cover the other's weaknesses.

**Composition.**
- `LexicalIndex` (`index/lexical.py`): BM25 over caption documents (`rank-bm25`, pure Python). The tokenizer
  reuses `koutil` (particle splitting + KO→EN synonyms). **Candidate selection is by token overlap, not score > 0** —
  Okapi IDF is 0 when a term appears in half the documents and negative when it is more common, so a term like "휴대폰" (Korean for "phone") that appears in nearly every caption
  would be eliminated under a score-based criterion (a pitfall that actually surfaced during debugging).
- `SearchService` (`core/services.py`): branches on `cfg.search_mode`.
  - `vector`: the original CLIP cosine.
  - `keyword`: BM25 only.
  - `hybrid` (default): fan out broadly on both arms and fuse the rankings with **RRF (k=60)**.
- **Score semantics are invariant**: `FolderHit.score` is *cosine in every mode*. The fused ranking goes in `fused_score`,
  and the match source goes in `match` (vector/keyword/both/graph, with gallery badges V·K·V+K·G). This is why a "score ≥" filter
  has the same meaning across all modes.
- **Cache and consistency**: the BM25 index is built lazily and rebuilt when `store.revision()` changes. After fusion, results are restored via `store.fetch`,
  so any id that has disappeared is naturally dropped (belt-and-suspenders).

### 3.4 GraphDB — object co-occurrence graph (`graph/*`)

**What.** From YOLO labels it builds an **image–object–folder** graph, enabling *structural* queries such as "images where a person and a motorcycle appear *together*"
or "objects that frequently appear alongside a phone" (a grain that semantic search cannot reach).

**Composition (one protocol, two backends).** Behind `GraphStore` in `graph/base.py`:
- `MemoryGraphStore` (default): pure-Python dict/set. Zero dependencies. Sufficient at the scale of a few thousand images.
- `KuzuGraphStore` (`[graph]` extra): an embedded GraphDB (kuzu), using **Cypher**. Like SQLite, it has no server.
- `registry.build_graph_store` looks at `cfg.graph_backend`, *lazily imports* kuzu, and falls back to memory if it fails.
- `builder.py` walks the dataset labels to produce `GraphRecord(image_path, folder, [class names])`.

**Philosophy and pitfalls.**
- *Joins between the graph and the index must always use `image_id()`*: the graph holds the on-disk original path (Windows backslashes, mixed case)
  as the node key, while the Chroma index keys by `image_id()` (slash/lowercase normalized). So when turning a path the graph returns into
  a gallery row, convert it with `image_id()` and then `store.fetch` (never compare the original path directly).
- *Safe under root mismatch*: if the graph was built from a different/moved dataset (its ids do not match the index), the filter
  is **skipped** and the hybrid results are shown as-is (preventing empty results).
- *Thread safety*: the graph build runs on a worker thread while reads come from the query/GUI thread, so contention arises. Both backends
  are protected by an `RLock` (memory atomically swaps in the completed map), and kuzu uses one Database + **a Connection per call** +
  `close()` to release the directory lock.
- UI: the toolbar **Object Graph** dialog (`ui/graph_dialog.py`) — class list, co-occurrence table, AND filter → gallery.

### 3.5 RAG (`core/services.py` + backends)

**Composition.** Retrieval → the sLLM summarizes in Korean (Generation) using the result captions as context.
- Both query refinement (`chat.refine_query`) and summarization (`chat.summarize`) are wrapped in try/except so that **search does not die even if they fail**,
  and the real backend (`openai_compat`) falls back once more internally (a double safety net).
- In mock mode, rule-based refinement (`koutil.expand_query`) and template summarization reproduce the same flow deterministically.

**Philosophy.** Summarization is constrained by prompts (`prompts.SUMMARY_*`) to speak *only within the retrieved evidence* — suppressing hallucination.

### 3.6 Multi-agent / Agentic RAG (`core/agentic.py`)

**What.** A single query is processed as a small A2A graph of **plan → hybrid search → object graph filter → summarize**,
showing each step as a trace in the chat window (`🧭 Plan / 🔍 Hybrid search / 🕸 Graph filter / ✍ Summary`).

**Composition.**
- `AgenticSearch(service, graph, chat, cfg)` — `run(query, step_cb)` streams a trace via `step_cb` at each step.
- **Planner**: decomposes the query into `Plan(semantic, required_objects, excluded_objects)`.
  - Real LLM: `chat.plan()` (vLLM, JSON output) is **called optionally via `getattr`** → so backends without `plan()` still work.
  - Fallback: `rule_based_plan` (koutil) — treats class names appearing in the query as required/excluded (detecting both English negators like "without"/"no"/"except" and Korean ones like 없는/말고/빼고).
    It is deterministic, so the full pipeline is reproduced even in mock mode.
- **Graph filter**: narrows down with `graph.images_with_all(required)` and reconciles with the index via `image_id()`. Missing co-occurring
  images are backfilled but assigned a **real cosine** to preserve score semantics. Excluded objects are dropped.
- **Threading**: it runs on the query worker thread, and `step_cb` is connected to the worker's `trace` signal (`workers/query_worker.py`),
  marshalling onto the GUI thread (no direct widget access).

**Philosophy.** The goal of "traverse a graph with A2A and mutually refine" is implemented as an *app feature* — the planner combines semantic search (soft)
with structural constraints (hard). If any step comes up empty (graph not built / mismatch), only that step is skipped and the rest proceeds.

### 3.7 vLLM — real-mode LLM/VLM (`backends/openai_compat.py`, `backends/prompts.py`)

**Composition.** vLLM is OpenAI-compatible HTTP, so it is called using only `httpx` (the client needs no `vllm` package). The same client also works
against compatible servers like Ollama. The constructor pings `/models` with a short timeout → a dead endpoint falls back cleanly at startup.
- `OpenAICompatVLMCaptioner.caption` — Korean captions from a base64 image.
- `OpenAICompatChatLLM.refine_query / summarize / plan` — `plan()` **does not trust** the JSON: it strips code fences +
  does `json.loads` in try/except, returning an empty dict on failure (→ the caller falls back to the rule-based planner).
- Check: `scripts/check_vllm.py` outputs the connect/refine/summarize/plan/(optionally)caption round-trips as PASS/FAIL.

**Philosophy.** Every real-mode call must *always be able to fall back to mock/rule*. A network, JSON, or model hiccup must never kill the whole search.

---

## 4. Data Flow

### 4.1 Indexing (`index/indexer.py`)
```
dataset walk → (per-image or folder representative) → caption (YOLO label/sidecar, else VLM) →
  embedding (batches of 32) → ChromaDB upsert → thumbnail prewarm (separate pool) → BuildReport
```
- Incremental: skip records that have not changed, keyed by `mtime(image)|mtime(label)`. Records that have disappeared from disk are pruned.
- After a build: invalidate the BM25 cache + rebuild the object graph (worker).

### 4.2 Search (`core/services.SearchService.query`)
```
query → refine(chat) → [mode]
  vector : embed_text → store.query(cosine top-k)
  keyword: BM25(lexical) → fetch → score = stored vector · query vector (cosine)
  hybrid : (vector top-N) + (BM25 top-N) → RRF fusion → fetch → assign cosine score
→ FolderHit[] → (RAG summary) → gallery (apply score≥ filter)
```

### 4.3 Agent (`core/agentic.AgenticSearch.run`)
```
query → plan(LLM/rule) → hybrid search(fan-out) → graph filter(required AND / excluded) → summarize
            └─ each step is traced in the chat
```

---

## 5. Threading Model (summary)

| Task | Thread | Connection method |
|---|---|---|
| Indexing / caption refresh | `IndexWorker` (QThread) | progress·log·completion signals (queued) |
| Search / agent | `QueryWorker` (QThread) | `finished`/`error`/`trace` (queued) |
| Model loading | `PreloadWorker` (QThread) | show the window first → swap the backend after completion |
| Graph build | `GraphBuildWorker` (QThread) | automatic after indexing |
| Thumbnails | `QThreadPool` + `QRunnable` | decode off-thread, QPixmap on the GUI |
| Index cancel | GUI→worker bool | **`Qt.DirectConnection`** (avoids starving the event loop) |

Shutdown (`closeEvent`) drains all runners and releases the kuzu lock with `graph.close()`.

---

## 6. Configuration and Paths

- `config.py` `AppConfig` (JSON): `search_mode`/`agentic_enabled`/`graph_backend`/`score_threshold`/`top_k`/
  `index_granularity`/`class_names`, etc. Runtime-only state (`degraded`) is not saved. Unknown keys are ignored on load (forward compatibility).
- `paths.py` `AppPaths`: `config.json`·`class_names.yaml`·`history.json`·`index_meta.json`·`chroma/`·`graph/`·
  `thumbs/`·`logs/`. Tests are isolated via `IMGSEARCH_HOME`.

---

## 7. Extension Points

- **New embedder/captioner/chat backend**: implement the `backends/base.py` protocol → add a branch in `core/registry.py`.
- **New graph backend** (e.g., a real Neo4j server): implement `GraphStore` in `graph/base.py` → add it to `graph/registry.py`.
- **New search mode/fusion**: add a branch in `core/services._retrieve`.
- **New agent step**: add the step to `core/agentic.AgenticSearch.run` (one line of trace via `step_cb`).

---

## 8. Collection of Key Gotchas

1. **Chroma re-embedding**: missing `embeddings=` on `update()` → the default EF re-embeds at 384-dim. → `update_meta` puts it back.
2. **BM25 IDF=0**: for common terms Okapi IDF is 0/negative → they fail a score > 0 filter. → select candidates by **token overlap**.
3. **Graph↔index join**: always use `image_id()` (never compare the original path directly).
4. **Qt cancel**: since the worker event loop is starved, cancel uses a `DirectConnection` bool while the trace uses a queued connection.
5. **Graph concurrency**: build (worker) and read (query/GUI) are serialized with an `RLock`. kuzu uses one Database + a Connection per call + `close()`.
6. **Score semantics**: `score` = cosine in every mode. Graph-membership hits are excluded from the threshold filter rather than from cosine.
7. **Version pins**: `transformers<5`, `torch<2.7` (jina-clip remote-code / `float8` issues). kuzu/torch are *optional* extras.

---

*Last updated: v0.3.1. See [`../CHANGELOG.md`](../CHANGELOG.md) for the change history and [`../README.md`](../README.md) for usage.*
