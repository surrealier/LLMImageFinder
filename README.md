# imgsearch — 데이터셋 자연어 이미지 검색 (sLLM_)

이미지 데이터셋을 **한국어 자연어로 질문해서 검색**하는 로컬 데스크톱 앱입니다.

> 예) "밤에 오토바이가 주차되어 있는 이미지", "사람이 대로변에 돌아다니는 이미지",
> "불과 연기가 있는 이미지"

폴더별 **대표 이미지 한 장**(또는 이미지별 개별 레코드)을 그리드로 보여주고,
더블클릭하면 확대(확대/이동/←→ 탐색), ‘폴더 열기’로 탐색기에서 원본 위치를 엽니다.

- **GUI**: PySide6 + qtawesome (다크 테마, 비차단 인덱싱/검색, 백그라운드 모델 로딩)
- **벡터 DB**: ChromaDB (`PersistentClient`, cosine) — 리프 폴더당(또는 이미지당) 벡터 1개
- **하이브리드 검색**: 벡터(CLIP) + 키워드(BM25, `rank-bm25`)를 **RRF**로 융합 — 의미·정확 매칭 결합
- **GraphDB**: 객체 공출현 그래프 — 메모리(기본) 또는 **kuzu**(임베디드, Cypher, `[graph]` extra)
- **멀티에이전트(RAG)**: 계획→하이브리드 검색→그래프 필터→요약 A2A 파이프라인 (트레이스 표시)
- **임베딩(검색 핵심)**: `jinaai/jina-clip-v2` (다국어 CLIP, 한국어 텍스트→이미지 직접 검색, 배치 GPU 임베딩)
- **VLM 캡션 / sLLM 채팅**: Qwen2.5-VL (vLLM, OpenAI 호환 API) — 캡션 자동 추출 + 질의 정제·요약·계획(plan)
- **MOCK 모드**: ML 의존성 0으로도 **전체 기능이 즉시 동작** (하이브리드·그래프·에이전트 포함, 결정적)

### 주요 기능 (v0.3.0)

| 기능 | 사용법 |
|---|---|
| 검색 방식 전환 | 갤러리 헤더 **콤보** — 하이브리드 / 벡터 / 키워드 |
| 에이전트 검색 | 헤더 **에이전트** 체크 — 계획→검색→그래프 필터→요약, 단계 트레이스 표시 |
| 객체 그래프 | 툴바 **객체 그래프** — 공출현 + "선택 객체 모두 포함한 이미지 보기"(AND) |
| 검색어 히스토리 | 입력창에서 **↑/↓** |
| 비슷한 이미지 검색 | 타일 **우클릭 → 비슷한 이미지 검색** (저장 임베딩 재사용, 즉시) |
| YOLO 박스 오버레이 | 뷰어에서 **B** 또는 '라벨' 버튼 (클래스명·색상 표시) |
| 결과 순서 탐색 | 뷰어 ←/→가 검색 랭킹 순으로 이동, 점수·캡션 표시 |
| 결과 수 k / 점수 필터 | 갤러리 헤더의 **결과 수** · **점수 ≥** 컨트롤 |
| 결과 내보내기 | 헤더 **내보내기** → CSV 저장 / 경로 목록 복사 |
| 삭제 파일 정리 | 인덱스 업데이트 시 디스크에서 사라진 레코드 자동 삭제 |
| 인덱스 정보 | 툴바 **인덱스 정보** — 모델/단위/빌드 시각/경로 불일치 경고 |
| 단축키 | **F5** 인덱스 업데이트 · **Ctrl+L** 검색창 · **Enter** 뷰어 · **Ctrl+C** 경로 복사 · **Ctrl+E** 탐색기 |

---

## 1. 빠른 시작 (MOCK 모드 — 모델 설치 불필요)

[uv](https://docs.astral.sh/uv/)가 필요합니다(이미 설치됨). Python 3.12는 uv가 자동 설치합니다.

```powershell
# 의존성 설치 (PySide6, chromadb, pillow, numpy, httpx 등)
uv sync

# 앱 실행
uv run imgsearch
#  또는
uv run python -m imgsearch
```

처음에는 **MOCK 모드**로 실행되며 상단에 노란 배너가 표시됩니다. 이 상태에서도
인덱싱 → 검색 → 확대 → 폴더 열기까지 전체 흐름을 그대로 사용할 수 있습니다(결과는
결정적·재현 가능). 실제 의미 검색은 아래 *실제 모델 활성화*에서 켭니다.

### 예시 데이터로 바로 체험

데이터셋이 없다면 합성 샘플을 만들 수 있습니다.

```powershell
uv run imgsearch --make-sample .\sample_dataset
```

또는 앱 툴바의 **‘샘플 데이터셋 생성’** 버튼을 누르면 폴더를 만들고 바로 인덱싱까지 진행합니다.
그런 다음 채팅창에 `불과 연기가 있는 이미지` 처럼 입력해 보세요.

---

## 2. 내 데이터셋 사용하기

1. 툴바 **설정** → **데이터셋 루트** 에서 폴더 선택.
2. 툴바 **인덱스 빌드/업데이트** 클릭 → 진행 표시줄이 끝나면 검색 가능.
3. 채팅창(`sLLM_`)에 찾고 싶은 장면을 한국어로 설명.

### 데이터셋 폴더 구조

- 이미지가 들어 있는 **모든 폴더가 하나의 "리프 폴더"** 로 취급되어 **대표 이미지 1장**으로 요약됩니다.
- 한 리프 폴더 안에 같은 장면의 사진이 여러 장 있어도 괜찮습니다(대표 1장만 노출, 더블클릭 시 전부 탐색).
- **사이드카 텍스트**(선택): 이미지와 **같은 이름**의 `.txt` / `.caption` / `.json` 파일이 있으면
  캡션/RAG 컨텍스트로 사용합니다. 폴더 단위 `caption.txt` / `readme.txt` 도 인식합니다.

```
dataset_root/
├─ 도심_야간/
│  └─ 오토바이_주차/            ← 리프 폴더 (대표 1장)
│     ├─ frame_000.jpg
│     ├─ frame_000.txt         ← (선택) 같은 이름 사이드카
│     ├─ frame_001.jpg
│     └─ ...
└─ 사고/
   └─ 화재_연기/
      ├─ img01.jpg ...
```

> 데이터셋 폴더에는 아무것도 기록하지 않습니다. 인덱스/썸네일/설정/로그는 모두
> 사용자 앱 데이터 폴더(`%LOCALAPPDATA%\MarkAny\ImgSearch\`)에 저장됩니다.

### YOLO 라벨 데이터셋과 클래스 이름

사이드카 `.txt`가 YOLO 탐지 라벨(`class_id cx cy w h`)인 데이터셋은 자동 인식됩니다
(라벨 숫자가 캡션에 새지 않음). 이때 **설정 → 색인 단위**를 `이미지별 개별 색인`으로
두면 이미지 한 장이 한 레코드가 되고, 캡션은 라벨의 객체 요약으로 생성됩니다.

클래스 ID → 이름 매핑은 **사용자가 직접 입력**합니다 (데이터셋 폴더는 수정하지 않음):

- **인덱스 빌드 시**: 이름 없는 클래스 ID가 발견되면 입력 테이블이 자동으로 뜹니다.
- **툴바 ‘클래스 이름’ 버튼**: 언제든 수정. 저장하면 기존 인덱스의 캡션을
  **재임베딩 없이 몇 초 만에 갱신**할지 물어봅니다.
- **CLI**: `uv run python scripts/set_class_names.py 0=사람 1=오토바이` (자동 캡션 갱신,
  `--show`로 현재 매핑 확인).
- 입력값은 `%LOCALAPPDATA%\MarkAny\ImgSearch\class_names.yaml`(ultralytics `names:` 형식)
  로 저장되며, 이 파일을 직접 편집하면 다음 실행 때 반영됩니다.

---

## 2.5 검색 방식 · 객체 그래프 · 에이전트 검색 (v0.3.0)

### 하이브리드 검색
갤러리 헤더의 **검색 방식** 콤보에서 고릅니다(설정 자동 저장).
- **하이브리드**(기본): 벡터(CLIP 의미)와 키워드(BM25, 캡션/라벨 정확 매칭)를 RRF로 융합.
- **벡터**: 의미 검색만. **키워드**: BM25만(클래스명 등 정확한 단어에 강함).

점수 배지는 어느 모드에서나 코사인 유사도이고, 정렬은 융합 순위를 따릅니다.

### 객체 그래프 (GraphDB)
툴바 **객체 그래프**를 열면 YOLO 라벨에서 만든 공출현 그래프를 탐색할 수 있습니다.
클래스별 이미지 수, 선택한 객체와 **함께 나타나는 객체**, 그리고 여러 객체를 체크해
**모두 포함한 이미지만** 갤러리에 표시합니다. 기본은 메모리 백엔드(의존성 없음)이며,
**설정 → 객체 그래프 백엔드**에서 `kuzu`(임베디드 GraphDB)로 바꿀 수 있습니다:

```powershell
uv sync --extra graph      # kuzu 설치 (없으면 자동으로 메모리 그래프로 폴백)
```

### 에이전트 검색 (멀티에이전트 RAG)
헤더 **에이전트**를 켜면 질의가 **계획 → 하이브리드 검색 → 객체 그래프 필터 → 요약**
파이프라인을 거치고, 각 단계가 채팅창에 트레이스로 표시됩니다. 예: "사람과 오토바이가
함께 있는 이미지"는 플래너가 `필수=[사람, 오토바이]`로 분해 → 그래프가 두 객체를 모두
포함한 이미지로 좁힙니다. vLLM이 연결되면 계획을 LLM(`plan()`)이 세우고, 아니면 규칙
기반으로 결정적 동작합니다(MOCK 모드 포함).

---

## 3. 실제 모델 활성화 (RTX 3070 8GB 기준)

MOCK은 데모/오프라인용입니다. 실제 의미 검색·캡션·요약을 쓰려면 두 가지를 켭니다.

### (a) 임베딩: jina-clip-v2 (로컬, in-process)

```powershell
uv sync --extra embed     # torch(CUDA) + sentence-transformers + einops 설치
```

**설정** → **임베딩** → 백엔드 `jina-clip`, 모델 `jinaai/jina-clip-v2`, 디바이스 `auto`(또는 `cuda`/`cpu`).
설정을 바꾸면 임베딩 공간이 달라지므로 **전체 재빌드**를 권장(앱이 자동으로 물어봅니다).

### (b) 캡션 + 채팅: Qwen2.5-VL on vLLM (OpenAI 호환 API)

vLLM은 Windows 네이티브 실행이 어렵습니다 → **WSL2 / Docker / 다른 호스트**에서 서버를 띄우고
앱은 그 **Base URL** 만 가리키면 됩니다.

```bash
# (WSL2 또는 Linux/Docker)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
    --port 8000 --max-model-len 8192
# -> http://localhost:8000/v1 에 OpenAI 호환 엔드포인트 제공
```

**설정** → **캡션(VLM)** 과 **채팅 sLLM** 에서 백엔드 `vllm`, Base URL `http://localhost:8000/v1`,
모델 `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` 입력. (하나의 Qwen2.5-VL 모델이 캡션·채팅·에이전트 plan을 모두 처리)

전환 전에 엔드포인트를 점검할 수 있습니다:

```powershell
uv run python scripts/check_vllm.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-3B-Instruct
# 연결 / refine / summarize / plan / (선택)caption 라운드트립을 PASS/FAIL로 출력
```

> 엔드포인트가 응답하지 않으면 자동으로 MOCK으로 폴백하고 배너에 사유를 표시합니다.
> Ollama 등 다른 OpenAI 호환 서버도 동일하게 사용할 수 있습니다.

### VRAM 메모 (8GB)
- jina-clip-v2: in-process ~2–3GB (필요 시 `cpu`로 전환 가능, 느려짐).
- Qwen2.5-VL: vLLM 별도 프로세스(WSL/Docker)에서 관리 → 인프로세스 메모리와 분리.
  AWQ(int4) 빌드 권장(~4GB). 캡션은 인덱싱 시점에만 호출됩니다.

---

## 4. 동작 방식

1. **인덱싱**: 데이터셋을 순회하며 리프 폴더마다 → 대표 이미지 선정(임베딩 중심값/`centroid`,
   모델 없으면 dHash medoid) → VLM 한국어 캡션 → CLIP 이미지 임베딩 → ChromaDB upsert.
   (mtime 기반 증분 업데이트, 진행률/취소 지원, UI 비차단)
2. **검색**: 한국어 질의를 sLLM이 정제 → (하이브리드) CLIP 텍스트 임베딩 + BM25 키워드 →
   RRF 융합 → ChromaDB top-k → 폴더별/이미지별 그리드 → 상위 결과 근거로 sLLM이 RAG 요약.
3. **에이전트**: 플래너가 질의를 의미+객체 제약으로 분해 → 하이브리드 검색 → 객체 그래프로
   필수/제외 객체 필터 → 요약. 단계별 트레이스를 채팅에 표시.
4. **표시**: 대표 이미지 타일(썸네일+캡션+폴더+유사도). 더블클릭 확대, ‘폴더 열기’로 탐색기.

---

## 5. 개발 / 테스트

```powershell
uv run --extra dev pytest               # 단위/통합 테스트 (mock, 모델 불필요)
uv run --extra dev --extra graph pytest # kuzu GraphDB 동등성 테스트까지 포함
uv run python scripts/smoke.py          # 헤드리스 파이프라인 점검
uv run python scripts/ui_smoke.py       # 오프스크린 GUI 점검
```

### 프로젝트 구조

```
imgsearch/
  app.py / __main__.py        진입점·테마
  config.py / paths.py        설정(JSON) · 앱 데이터 경로
  koutil.py                   한국어 토크나이즈/KO→EN 사전 (mock·BM25 공용)
  sample_data.py              합성 데이터셋 생성기
  thumbs.py                   썸네일 캐시 (PIL, 스레드 안전)
  core/      models · services(SearchService: 하이브리드·RRF) · agentic(A2A) · registry
  backends/  base(Protocol) · mock · jina_clip · openai_compat(plan 포함) · prompts
  index/     walker · pairing · repr_select · indexer · lexical(BM25) · labels
  store/     chroma_store (search_vector · all_documents · fetch · revision)
  graph/     base(Protocol) · memory_graph · kuzu_graph · builder · registry
  ui/        main_window · chat_widget · results_gallery · gallery_delegate
             image_viewer · settings_dialog · class_names_dialog · graph_dialog
             index_info_dialog · mock_banner · icons · osutil
  workers/   qworker · index_worker · query_worker(trace) · thumb_worker
             preload_worker · graph_worker
tests/       walker · pairing · repr_select · mock_embedder · chroma_store · labels
             query_refine · indexer_query · class_names · config · services
             build_pipeline · hybrid · graph · agentic
scripts/     smoke · ui_smoke · build_index · set_class_names · check_vllm · …
```

## 6. 문제 해결

- **노란 배너(MOCK)가 계속 보임**: 의도된 폴백입니다. `[embed]` 설치 + 설정에서 백엔드를
  `jina-clip`/`vllm`로 바꾸고, vLLM 서버가 떠 있는지 확인하세요(배너에 사유 표시).
- **검색 결과가 이상함(MOCK)**: MOCK은 폴더명/사이드카 텍스트 키워드 매칭에 의존합니다.
  실제 의미 검색은 jina-clip 백엔드에서 동작합니다.
- **인덱싱이 느림**: 캡션(VLM)을 끄거나(설정), 임베딩 디바이스를 `cuda`로 두세요.
  증분 업데이트는 변경된 폴더만 다시 처리합니다.
- **키워드/하이브리드 결과가 비어 있음**: BM25는 캡션의 *정확한 단어*에 매칭됩니다
  (예: 캡션이 "휴대폰"이면 "스마트폰"으로는 안 잡힘). 의미 검색은 벡터/하이브리드가 담당.
- **객체 그래프가 비어 있음**: YOLO 라벨이 있는 데이터셋을 색인해야 채워집니다. 인덱스
  빌드 후 자동 구축되며, ‘객체 그래프’를 처음 열 때도 만들어집니다.
- **에이전트 검색이 그래프 필터를 건너뜀**: 그래프가 아직 안 만들어졌거나 현재 인덱스와
  경로가 다를 때입니다(전체 재빌드 권장). 이 경우 하이브리드 결과를 그대로 보여줍니다.
- **로그**: `%LOCALAPPDATA%\MarkAny\ImgSearch\logs\imgsearch.log`
