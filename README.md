# imgsearch — 데이터셋 자연어 이미지 검색 (sLLM_)

이미지 데이터셋을 **한국어 자연어로 질문해서 검색**하는 로컬 데스크톱 앱입니다.

> 예) "밤에 오토바이가 주차되어 있는 이미지", "사람이 대로변에 돌아다니는 이미지",
> "불과 연기가 있는 이미지"

폴더별 **대표 이미지 한 장**을 그리드로 보여주고, 더블클릭하면 확대(확대/이동/←→ 탐색),
‘폴더 열기’로 탐색기에서 원본 위치를 엽니다.

- **GUI**: PySide6 + qtawesome (다크 테마, 비차단 인덱싱/검색)
- **벡터 DB**: ChromaDB (`PersistentClient`, cosine) — 리프 폴더당 벡터 1개
- **임베딩(검색 핵심)**: `jinaai/jina-clip-v2` (다국어 CLIP, 한국어 텍스트→이미지 직접 검색)
- **VLM 캡션**: Qwen2.5-VL (vLLM, OpenAI 호환 API) — 대표 이미지의 한국어 설명 자동 추출
- **sLLM 채팅**: Qwen (vLLM) — 질의 정제 + 결과 RAG 요약
- **MOCK 모드**: ML 의존성 0으로도 **전체 기능이 즉시 동작** (결정적 임베딩 + 규칙 기반 한국어 처리)

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
모델 `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` 입력. (하나의 Qwen2.5-VL 모델이 캡션과 채팅을 모두 처리)

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
2. **검색**: 한국어 질의를 sLLM이 정제 → CLIP 텍스트 임베딩 → ChromaDB cosine top-k →
   폴더별 대표 이미지 그리드 → 상위 결과 캡션을 근거로 sLLM이 한국어 요약(RAG).
3. **표시**: 대표 이미지 타일(썸네일+캡션+폴더+유사도). 더블클릭 확대, ‘폴더 열기’로 탐색기.

---

## 5. 개발 / 테스트

```powershell
uv run --extra dev pytest          # 단위/통합 테스트 (mock, 모델 불필요)
uv run python scripts/smoke.py     # 헤드리스 파이프라인 점검
uv run python scripts/ui_smoke.py  # 오프스크린 GUI 점검
```

### 프로젝트 구조

```
imgsearch/
  app.py / __main__.py        진입점·테마
  config.py / paths.py        설정(JSON) · 앱 데이터 경로
  koutil.py                   한국어 토크나이즈/KO→EN 사전 (mock)
  sample_data.py              합성 데이터셋 생성기
  thumbs.py                   썸네일 캐시 (PIL, 스레드 안전)
  core/      models · services(SearchService) · registry(백엔드 팩토리)
  backends/  base(Protocol) · mock · jina_clip · openai_compat · prompts
  index/     walker · pairing · repr_select · indexer
  store/     chroma_store
  ui/        main_window · chat_widget · results_gallery · gallery_delegate
             image_viewer · settings_dialog · mock_banner · icons · osutil
  workers/   qworker · index_worker · query_worker · thumb_worker
tests/       walker · pairing · repr_select · mock_embedder · chroma_store
             query_refine · indexer_query (통합)
```

## 6. 문제 해결

- **노란 배너(MOCK)가 계속 보임**: 의도된 폴백입니다. `[embed]` 설치 + 설정에서 백엔드를
  `jina-clip`/`vllm`로 바꾸고, vLLM 서버가 떠 있는지 확인하세요(배너에 사유 표시).
- **검색 결과가 이상함(MOCK)**: MOCK은 폴더명/사이드카 텍스트 키워드 매칭에 의존합니다.
  실제 의미 검색은 jina-clip 백엔드에서 동작합니다.
- **인덱싱이 느림**: 캡션(VLM)을 끄거나(설정), 임베딩 디바이스를 `cuda`로 두세요.
  증분 업데이트는 변경된 폴더만 다시 처리합니다.
- **로그**: `%LOCALAPPDATA%\MarkAny\ImgSearch\logs\imgsearch.log`
