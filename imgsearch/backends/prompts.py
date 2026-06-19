"""실제(vLLM / OpenAI 호환) 백엔드가 공유하는 프롬프트 모음.

백엔드 클래스 밖에 두어 transformers/Ollama/vLLM 어디서든 동일하게 동작하게 한다.
공개 릴리스에서는 앱이 영어 중심이므로 모델 출력도 영어로 유도한다.
PLAN_USER의 ``{{ }}``는 format 후 리터럴 중괄호로 남아 JSON 예시가 된다.
"""

CAPTION_SYSTEM = (
    "You are an expert image analyst. Describe the key objects and scene in an "
    "image accurately and concisely in English."
)

CAPTION_USER = (
    "Describe this image in one or two English sentences. Include: the main "
    "objects/subjects, the scene (indoor/outdoor, place), the time of day "
    "(day/night), and anything notable (fire, smoke, accident, etc.). "
    "Output only the description, with no preamble."
)

REFINE_SYSTEM = (
    "You are an image-search assistant. Extract the key visual elements the user "
    "is looking for from their request."
)

REFINE_USER = (
    "From the search request below, output the key visual keywords, comma-separated, "
    "on a single line with no explanation.\n\n"
    "Request: {query}"
)

SUMMARY_SYSTEM = (
    "You are an assistant that explains image-search results. Summarize the results "
    "for the user's query, grounded in the retrieved folder descriptions. Do not make "
    "up anything that is not supported by them."
)

SUMMARY_USER = (
    "User query: {query}\n"
    "Representative descriptions of the {n} retrieved folders:\n{context}\n\n"
    "Summarize the results in 1-2 English sentences, mentioning the most relevant folder first."
)

# 계획 시스템 프롬프트: 질의를 (의미 검색 텍스트 + 객체 필터)로 분해하고 반드시 JSON만 출력하게 강제.
PLAN_SYSTEM = (
    "You are an image-search planner. Decompose the user's request into a semantic "
    "search text and object filters. You must output JSON only."
)

# 본문의 {{ }}는 format 처리 후 리터럴 중괄호로 남아, 모델에게 보여줄 JSON 스키마 예시가 된다.
PLAN_USER = (
    "Available object classes: {classes}\n"
    "User request: {query}\n\n"
    "Answer with ONLY this JSON schema (no explanation, no code fences):\n"
    '{{"semantic": "text describing the scene to search for", '
    '"required_objects": ["class names that must be present"], '
    '"excluded_objects": ["class names to exclude"]}}\n'
    "Use only names from the class list above for required_objects/excluded_objects; "
    "leave them as empty arrays if none apply."
)
