"""Korean prompts shared by the real (vLLM / OpenAI-compatible) backends.

Kept out of the backend classes so transformers/Ollama/vLLM all behave identically.
"""

CAPTION_SYSTEM = (
    "당신은 이미지 분석 전문가입니다. 이미지를 보고 핵심 객체와 장면을 한국어로 "
    "정확하고 간결하게 설명합니다."
)

CAPTION_USER = (
    "이 이미지를 한국어로 한두 문장으로 설명하세요. 다음을 포함하세요: "
    "주요 객체/대상, 장면(실내/실외·장소), 시간대(주간/야간), 특이사항(불·연기·사고 등). "
    "사족이나 머리말 없이 설명 문장만 출력하세요."
)

REFINE_SYSTEM = (
    "당신은 이미지 검색 보조자입니다. 사용자의 한국어 요청에서 찾고자 하는 이미지의 "
    "핵심 시각 요소를 추출합니다."
)

REFINE_USER = (
    "다음 검색 요청에서 찾으려는 이미지의 핵심 키워드를 쉼표로 구분해 출력하세요. "
    "한국어 키워드를 우선하되 필요하면 영어를 병기하세요. 설명 없이 키워드만 한 줄로.\n\n"
    "요청: {query}"
)

SUMMARY_SYSTEM = (
    "당신은 이미지 검색 결과를 설명하는 보조자입니다. 검색된 폴더 설명을 근거로 "
    "사용자 질의에 맞는 결과를 한국어로 요약합니다. 근거에 없는 내용은 지어내지 마세요."
)

SUMMARY_USER = (
    "사용자 질의: {query}\n"
    "검색된 폴더 {n}개의 대표 설명:\n{context}\n\n"
    "위 결과를 1~2문장 한국어로 요약하세요. 가장 관련성 높은 폴더를 우선 언급하세요."
)

# --- agentic planner: decompose a query into a structured retrieval plan ---
PLAN_SYSTEM = (
    "당신은 이미지 검색 플래너입니다. 사용자의 한국어 요청을 의미 검색 텍스트와 "
    "객체 필터로 분해합니다. 반드시 JSON만 출력합니다."
)

PLAN_USER = (
    "사용 가능한 객체 클래스: {classes}\n"
    "사용자 요청: {query}\n\n"
    "다음 JSON 스키마로만 답하세요(설명·코드펜스 금지):\n"
    '{{"semantic": "장면을 묘사하는 검색 텍스트", '
    '"required_objects": ["반드시 포함할 클래스명"], '
    '"excluded_objects": ["제외할 클래스명"]}}\n'
    "required_objects/excluded_objects의 값은 위 클래스 목록에 있는 이름만 사용하고, "
    "해당 없으면 빈 배열로 두세요."
)
