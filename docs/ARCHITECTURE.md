# 아키텍처와 기술 철학

> `imgsearch` — 한국어 자연어로 이미지 데이터셋을 검색하는 로컬 PySide6 데스크톱 앱.
> 이 문서는 앱에 들어간 핵심 기술(**RAG · VectorDB · 하이브리드 검색 · GraphDB · 멀티에이전트 · vLLM**)이
> 코드에서 *어떻게 구성*되어 있고 *왜 그렇게* 설계했는지를 정리합니다.

---

## 0. 한눈에

| 기술 | 한 줄 정의 | 사는 곳(주요 파일) |
|---|---|---|
| **VectorDB** | 이미지 임베딩을 코사인 공간에 저장·검색 | `store/chroma_store.py` (ChromaDB) |
| **임베딩** | 이미지·텍스트를 같은 512차원 공간으로 | `backends/jina_clip.py`(실제), `backends/mock.py`(목) |
| **하이브리드 검색** | 벡터(의미)+키워드(BM25)를 RRF로 융합 | `index/lexical.py` + `core/services.py` |
| **GraphDB** | 객체 공출현 그래프(이미지–객체–폴더) | `graph/*` (메모리/kuzu) |
| **RAG** | 검색 결과를 근거로 한국어 요약 | `core/services.py` + `backends/openai_compat.py` |
| **멀티에이전트** | 계획→검색→그래프 필터→요약 A2A 파이프라인 | `core/agentic.py` |
| **vLLM** | 캡션·요약·질의정제·계획을 LLM이 수행 | `backends/openai_compat.py`, `backends/prompts.py` |

---

## 1. 다섯 가지 관통 철학

모든 기술 선택은 아래 다섯 원칙을 따릅니다. 코드를 읽을 때 "왜 이렇게 했지?"의 답은 대부분 여기에 있습니다.

### (1) Mock-first — ML 의존성 0으로도 전 기능이 즉시 동작
- `uv sync`만 하면(torch/transformers/kuzu 없이) **인덱싱·하이브리드 검색·그래프·에이전트까지 전부** 돌아갑니다.
- 실제 모델은 *추가 의존성*(`[embed]`, `[graph]`)과 *설정*으로만 활성화됩니다.
- 목(mock) 백엔드는 **결정적**입니다(`backends/mock.py`, `koutil.py`의 md5 해시 버킷). 그래서 테스트가 재현 가능합니다.
- 이유: 데모/오프라인/CI에서 GPU·서버 없이 전체 UX를 검증하고, 실제 백엔드는 같은 인터페이스 뒤에 끼워 넣기 위함.

### (2) 어댑터/프로토콜 경계 — 앱은 모델 라이브러리를 직접 모른다
- `backends/base.py`에 `Embedder`/`Captioner`/`ChatLLM` **Protocol**만 정의. 나머지 코드는 이 추상만 봅니다.
- 백엔드는 문자열(`"mock"`/`"jina-clip"`/`"vllm"`)로 선택되고, `core/registry.py`가 *지연 import + try/except*로
  실제 백엔드를 만들되 실패하면 목으로 **폴백**하고 사유를 배너에 띄웁니다(`cfg.mark_degraded`).
- 그래프도 동일: `graph/base.py`의 `GraphStore` 프로토콜 뒤에 메모리/kuzu 두 구현.

### (3) 데이터셋 불가침 — 앱은 사용자 데이터셋 폴더에 절대 쓰지 않는다
- 인덱스·썸네일·설정·로그·클래스이름·그래프 모두 **앱 데이터 폴더**(`%LOCALAPPDATA%\MarkAny\ImgSearch\`)에만 저장.
- 경로는 `paths.py`가 한 곳에서 관리(`AppPaths`). 테스트는 `IMGSEARCH_HOME` 환경변수로 격리.
- 클래스 이름조차 데이터셋에 두지 않고 앱 데이터의 `class_names.yaml`에 둡니다(사용자 입력 기반).

### (4) UI 비차단 — GUI 스레드는 그리기만 한다
- 무거운 작업(임베딩, 인덱싱, 모델 로딩, 그래프 빌드, 검색)은 전부 **QThread 워커**(`workers/*`)에서 실행.
- 워커는 `ThreadRunner`(`workers/qworker.py`)가 참조를 붙잡아 GC를 막고 시그널로만 GUI와 통신.
- 함정 메모: 워커 스레드는 `run()` 안에서 막혀 있어 이벤트 루프가 굶습니다. 그래서 **취소 플래그는
  `Qt.DirectConnection`**(GIL-원자적 bool)로, **에이전트 트레이스는 큐 연결**(위젯 접근이므로 GUI 스레드에서 실행)로 연결합니다.

### (5) 결정적·검증 가능 — 84개 테스트 + 오프스크린 UI 스모크
- 핵심 동작은 모두 헤드리스로 테스트(`tests/`). UI는 `scripts/ui_smoke.py`가 오프스크린으로 구동.
- 실제 모델 경로는 `scripts/build_index.py`/`check_vllm.py`로 라이브 검증.

---

## 2. 레이어 지도

```
                         ┌──────────────────────────────────────────────┐
  사용자 ◀─ 채팅/갤러리 ─▶ │  ui/  (main_window · chat_widget · gallery ·  │
                         │       image_viewer · *_dialog · delegate)     │
                         └───────────────┬───────────────────────────────┘
                            시그널/슬롯    │  (무거운 일은 워커로 위임)
                         ┌───────────────▼───────────────────────────────┐
                         │  workers/  (QThread: index · query · graph ·   │
                         │            preload · thumb)                    │
                         └───────────────┬───────────────────────────────┘
                                         │
        ┌────────────────────────────────▼────────────────────────────────┐
        │  core/   services(SearchService: 하이브리드·RAG) · agentic(A2A)  │
        │          registry(백엔드 팩토리) · models(dataclass)             │
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
   │ 횡단: config(설정) · paths(앱데이터 경로) · koutil(한국어/목 공용) ·  │
   │       thumbs(썸네일 캐시) · logging_setup                            │
   └──────────────────────────────────────────────────────────────────────┘
```

규칙: **위 레이어는 아래 레이어만 안다.** `core`는 Qt를 import하지 않습니다(그래서 헤드리스 테스트 가능).
`backends`/`graph`/`store`는 서로를 모릅니다(프로토콜로만 연결).

---

## 3. 기술별 정리

### 3.1 VectorDB — ChromaDB (`store/chroma_store.py`)

**구성.** `representatives` 컬렉션 1개, 코사인 공간(`hnsw:space=cosine`). 레코드 1개 =
(id, 임베딩, document=캡션, metadata). id는 경로의 sha1 해시라서 재색인이 **idempotent upsert**.

**철학.**
- *우리 임베딩만 쓴다*: Chroma의 기본 임베딩 함수(EF)를 절대 붙이지 않습니다. CLIP/목 벡터를 직접 넣습니다.
  - 함정(실제 버그였음): `update()`에 `embeddings=`를 주지 않으면 Chroma가 기본 ONNX MiniLM(384차원)으로
    document를 **재임베딩**해 공간을 오염시킵니다. → `update_meta`는 저장된 임베딩을 읽어 되돌려 넣습니다.
- *변경 감지용 revision 카운터*: `_rev`를 모든 쓰기(upsert/update_meta/delete/clear)에서 증가시킵니다.
  캡션만 바뀌는 `update_meta`는 레코드 수가 그대로라 `count()`로는 못 잡습니다 → 하이브리드 BM25 캐시는
  `revision()`(=`(count, _rev)`)으로 staleness를 판정해 재구축합니다.
- *증분 색인*: `existing_mtimes()`가 `id→mtime키`를 페이지 단위로 읽어, 안 바뀐 레코드는 건너뜁니다.

**핵심 메서드.** `upsert`, `update_meta`(재임베딩 방지), `query`(코사인 top-k → `FolderHit`),
`search_vector`(하이브리드용, **id를 반환**), `all_documents`(BM25 코퍼스), `fetch`(id→FolderHit),
`get_embedding`/`get_embeddings`(저장 벡터 재사용), `revision`.

### 3.2 임베딩 — jina-clip-v2 / 목 (`backends/jina_clip.py`, `backends/mock.py`)

**구성.** `jinaai/jina-clip-v2`(다국어 CLIP)를 `sentence-transformers`로 인프로세스 구동, 512차원 Matryoshka 절단.
한국어 텍스트→이미지 검색이 **번역 없이** 직접 됩니다. 목은 결정적 해시 + 약간의 실제 시각 신호(축소 그레이/히스토그램).

**철학.** 텍스트·이미지가 *한 공간*에 L2 정규화되어 들어가야 코사인이 의미를 가집니다. 정규화돼 있으므로
**내적 = 코사인**이고, 그래서 키워드 전용 히트의 점수도 `저장벡터·질의벡터`로 싸게 계산합니다(`services._retrieve`).

### 3.3 하이브리드 검색 — BM25 + 벡터 + RRF (`index/lexical.py`, `core/services.py`)

**왜.** 벡터 검색은 *의미*는 잘 잡지만 "휴대폰 후면 카메라" 같은 *정확한 용어*는 놓칠 수 있고, 그 반대도 마찬가지입니다.
두 신호를 합치면 서로의 약점을 가립니다.

**구성.**
- `LexicalIndex`(`index/lexical.py`): 캡션 document에 대한 BM25(`rank-bm25`, 순수 파이썬). 토크나이저는
  `koutil` 재사용(조사 분리 + KO→EN 동의어). **후보 선정은 점수>0이 아니라 토큰 겹침**으로 합니다 —
  Okapi IDF는 용어가 문서 절반에 있으면 0, 더 흔하면 음수라, "휴대폰"처럼 거의 모든 캡션에 있는 용어가
  점수 기준으로는 탈락하기 때문입니다(실제로 디버깅에서 드러난 함정).
- `SearchService`(`core/services.py`): `cfg.search_mode`로 분기.
  - `vector`: 기존 CLIP 코사인.
  - `keyword`: BM25만.
  - `hybrid`(기본): 두 팔을 각각 넓게(fan-out) 가져와 **RRF(k=60)**로 순위 융합.
- **점수 의미 불변**: `FolderHit.score`는 *어느 모드에서나 코사인*입니다. 융합 순위는 `fused_score`,
  매칭 출처는 `match`(vector/keyword/both/graph, 갤러리 배지 V·K·V+K·G)에 담습니다. 그래서 "점수 ≥" 필터가
  모든 모드에서 동일한 의미를 가집니다.
- **캐시·정합성**: BM25 인덱스는 지연 생성, `store.revision()`이 바뀌면 재구축. 융합 후 `store.fetch`로
  되살리므로 사라진 id는 자연히 탈락(belt-and-suspenders).

### 3.4 GraphDB — 객체 공출현 그래프 (`graph/*`)

**무엇.** YOLO 라벨에서 **이미지–객체–폴더** 그래프를 만들어, "사람과 오토바이가 *함께* 있는 이미지",
"휴대폰과 자주 같이 나오는 객체" 같은 *구조적* 질의를 가능하게 합니다(의미 검색이 못 하는 결).

**구성(한 프로토콜, 두 백엔드).** `graph/base.py`의 `GraphStore` 뒤에:
- `MemoryGraphStore`(기본): 순수 파이썬 dict/set. 의존성 0. 수천 장 규모에 충분.
- `KuzuGraphStore`(`[graph]` extra): 임베디드 GraphDB(kuzu), **Cypher** 사용. SQLite처럼 서버가 없습니다.
- `registry.build_graph_store`가 `cfg.graph_backend`를 보고 kuzu를 *지연 import*, 실패하면 메모리로 폴백.
- `builder.py`가 데이터셋 라벨을 걸어 `GraphRecord(image_path, folder, [class names])`를 생성.

**철학·함정.**
- *그래프와 인덱스의 조인은 반드시 `image_id()`로*: 그래프는 디스크 원본 경로(윈도 역슬래시·대소문자)를
  노드 키로 갖고, Chroma 인덱스는 `image_id()`(슬래시·소문자 정규화)로 키를 갖습니다. 그래서 그래프가 돌려준
  경로를 갤러리 행으로 바꿀 땐 `image_id()`로 변환해 `store.fetch`합니다(원본 경로 직접 비교 금지).
- *루트 불일치 안전*: 그래프가 다른/이동한 데이터셋에서 만들어졌으면(인덱스와 id가 안 맞으면) 필터를
  **건너뛰고** 하이브리드 결과를 그대로 보여줍니다(빈 결과 방지).
- *스레드 안전*: 그래프 빌드는 워커 스레드, 읽기는 질의/GUI 스레드라 경쟁이 생깁니다. 두 백엔드 모두
  `RLock`으로 보호하고(메모리는 완성된 맵을 원자적으로 swap), kuzu는 Database 1개 + **호출마다 Connection** +
  `close()`로 디렉터리 락 해제.
- UI: 툴바 **객체 그래프** 다이얼로그(`ui/graph_dialog.py`) — 클래스 목록·공출현 표·AND 필터 → 갤러리.

### 3.5 RAG (`core/services.py` + 백엔드)

**구성.** 검색(Retrieval) → 결과 캡션을 컨텍스트로 sLLM이 한국어 요약(Generation).
- 질의 정제(`chat.refine_query`)와 요약(`chat.summarize`)은 둘 다 **실패해도 검색이 죽지 않게** try/except로 감쌌고,
  실제 백엔드(`openai_compat`)도 내부에서 한 번 더 폴백합니다(이중 안전망).
- 목 모드에선 규칙 기반 정제(`koutil.expand_query`)와 템플릿 요약으로 같은 흐름을 결정적으로 재현.

**철학.** 요약은 *검색된 근거 안에서만* 말하도록 프롬프트(`prompts.SUMMARY_*`)로 제약 — 환각 억제.

### 3.6 멀티에이전트 / 에이전틱 RAG (`core/agentic.py`)

**무엇.** 한 번의 질의를 **계획(plan) → 하이브리드 검색 → 객체 그래프 필터 → 요약**의
작은 A2A 그래프로 처리하고, 각 단계를 채팅창에 트레이스로 보여줍니다(`🧭 계획 / 🔍 검색 / 🕸 그래프 필터 / ✍ 요약`).

**구성.**
- `AgenticSearch(service, graph, chat, cfg)` — `run(query, step_cb)`가 단계마다 `step_cb`로 트레이스를 흘립니다.
- **플래너**: 질의를 `Plan(semantic, required_objects, excluded_objects)`로 분해.
  - 실제 LLM: `chat.plan()`(vLLM, JSON 출력)을 **`getattr`로 선택적 호출** → 그래서 `plan()`이 없는 백엔드도 동작.
  - 폴백: `rule_based_plan`(koutil) — 질의에 등장하는 클래스명을 필수/제외(없는·말고·빼고 등 부정어 감지)로.
    결정적이라 목 모드에서도 전 파이프라인이 재현됩니다.
- **그래프 필터**: `graph.images_with_all(required)`로 좁히고, `image_id()`로 인덱스와 정합. 누락된 공출현
  이미지는 backfill하되 **실제 코사인**을 매겨 점수 의미를 유지. 제외 객체는 떨어뜨림.
- **스레딩**: 질의 워커 스레드에서 돌고, `step_cb`는 워커의 `trace` 시그널(`workers/query_worker.py`)에 연결되어
  GUI 스레드로 마샬됩니다(위젯 직접 접근 금지).

**철학.** "A2A로 그래프를 돌며 상호 개선"이라는 목표를 *앱 기능*으로 구현 — 의미 검색(soft)과 구조 제약(hard)을
플래너가 조합합니다. 어느 단계가 비어도(그래프 미구축·불일치) 그 단계만 건너뛰고 나머지로 진행.

### 3.7 vLLM — 실모드 LLM/VLM (`backends/openai_compat.py`, `backends/prompts.py`)

**구성.** vLLM은 OpenAI 호환 HTTP라 `httpx`로만 호출합니다(클라이언트에 `vllm` 패키지 불필요). 같은 클라이언트가
Ollama 등 호환 서버에도 통함. 생성자에서 `/models`를 짧은 타임아웃으로 핑 → 죽은 엔드포인트는 시작 시 깔끔히 폴백.
- `OpenAICompatVLMCaptioner.caption` — base64 이미지로 한국어 캡션.
- `OpenAICompatChatLLM.refine_query / summarize / plan` — `plan()`은 JSON을 **신뢰하지 않고** 코드펜스 제거 +
  `json.loads` try/except, 실패 시 빈 dict(→ 호출부가 규칙 플래너로 폴백).
- 점검: `scripts/check_vllm.py`가 연결·refine·summarize·plan·(선택)caption 라운드트립을 PASS/FAIL로 출력.

**철학.** 모든 실모드 호출은 *언제나 목/규칙 폴백 가능*해야 합니다. 네트워크·JSON·모델 변덕이 검색 전체를 죽이면 안 됩니다.

---

## 4. 데이터 흐름

### 4.1 인덱싱 (`index/indexer.py`)
```
데이터셋 walk → (이미지 단위 or 폴더 대표) → 캡션(YOLO 라벨/사이드카, 없으면 VLM) →
  임베딩(배치 32장) → ChromaDB upsert → 썸네일 프리워밍(별도 풀) → BuildReport
```
- 증분: `mtime(이미지)|mtime(라벨)` 키로 안 바뀐 레코드 skip. 디스크에서 사라진 레코드는 prune.
- 빌드 후 BM25 캐시 무효화 + 객체 그래프 재빌드(워커).

### 4.2 검색 (`core/services.SearchService.query`)
```
질의 → 정제(chat) → [모드]
  vector : embed_text → store.query(코사인 top-k)
  keyword: BM25(lexical) → fetch → 점수는 저장벡터·질의벡터로 코사인
  hybrid : (벡터 top-N) + (BM25 top-N) → RRF 융합 → fetch → 코사인 점수 부여
→ FolderHit[] → (요약 RAG) → 갤러리(점수≥필터 적용)
```

### 4.3 에이전트 (`core/agentic.AgenticSearch.run`)
```
질의 → 계획(plan/규칙) → 하이브리드 검색(fan-out) → 그래프 필터(필수 AND / 제외) → 요약
            └─ 각 단계 트레이스를 채팅에 표시
```

---

## 5. 스레딩 모델 (요약)

| 작업 | 스레드 | 연결 방식 |
|---|---|---|
| 인덱싱/캡션갱신 | `IndexWorker` (QThread) | 진행률·로그·완료 시그널(큐) |
| 검색/에이전트 | `QueryWorker` (QThread) | `finished`/`error`/`trace`(큐) |
| 모델 로딩 | `PreloadWorker` (QThread) | 창 먼저 표시 → 완료 후 백엔드 교체 |
| 그래프 빌드 | `GraphBuildWorker` (QThread) | 인덱스 후 자동 |
| 썸네일 | `QThreadPool` + `QRunnable` | 디코드 오프스레드, QPixmap은 GUI에서 |
| 인덱스 취소 | GUI→워커 bool | **`Qt.DirectConnection`**(이벤트 루프 굶음 회피) |

종료(`closeEvent`)는 모든 러너를 드레인하고 `graph.close()`로 kuzu 락을 해제합니다.

---

## 6. 설정·경로

- `config.py` `AppConfig`(JSON): `search_mode`/`agentic_enabled`/`graph_backend`/`score_threshold`/`top_k`/
  `index_granularity`/`class_names` 등. 런타임 전용(`degraded`)은 저장하지 않음. 알 수 없는 키는 로드 시 무시(전방호환).
- `paths.py` `AppPaths`: `config.json`·`class_names.yaml`·`history.json`·`index_meta.json`·`chroma/`·`graph/`·
  `thumbs/`·`logs/`. `IMGSEARCH_HOME`로 테스트 격리.

---

## 7. 확장 포인트

- **새 임베더/캡셔너/챗 백엔드**: `backends/base.py` 프로토콜 구현 → `core/registry.py`에 분기 추가.
- **새 그래프 백엔드**(예: 실제 Neo4j 서버): `graph/base.py` `GraphStore` 구현 → `graph/registry.py`에 추가.
- **새 검색 모드/융합**: `core/services._retrieve`에 분기.
- **새 에이전트 단계**: `core/agentic.AgenticSearch.run`에 단계 추가(트레이스 `step_cb` 한 줄).

---

## 8. 핵심 함정(gotchas) 모음

1. **Chroma 재임베딩**: `update()`에 `embeddings=` 누락 → 기본 EF가 384차원으로 재임베딩. → `update_meta`가 되돌려 넣음.
2. **BM25 IDF=0**: 흔한 용어는 Okapi IDF가 0/음수 → 점수>0 필터로는 탈락. → **토큰 겹침**으로 후보 선정.
3. **그래프↔인덱스 조인**: 반드시 `image_id()`로(원본 경로 직접 비교 금지).
4. **Qt 취소**: 워커 이벤트 루프가 굶으므로 취소는 `DirectConnection` bool, 트레이스는 큐 연결.
5. **그래프 동시성**: 빌드(워커)와 읽기(질의/GUI)는 `RLock`으로 직렬화. kuzu는 Database 1개 + 호출별 Connection + `close()`.
6. **점수 의미**: 모든 모드에서 `score`=코사인. 그래프 멤버십 히트는 코사인이 아니라 임계값 필터에서 제외.
7. **버전 핀**: `transformers<5`, `torch<2.7`(jina-clip 원격코드/`float8` 이슈). kuzu/torch는 *선택* extra.

---

*최종 업데이트: v0.3.0. 변경 이력은 [`../CHANGELOG.md`], 사용법은 [`../README.md`] 참고.*
