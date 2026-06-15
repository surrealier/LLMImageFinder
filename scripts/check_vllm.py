"""Validate a real vLLM (OpenAI-compatible) endpoint end to end.

Exercises the full chat/VLM round-trip the app uses: connectivity, refine_query,
summarize, the agentic plan() JSON, and (optionally) an image caption. Prints a clear
PASS/FAIL per step so you can confirm the real backend before switching the app to it.

Usage:
  uv run python scripts/check_vllm.py                         # use saved config
  uv run python scripts/check_vllm.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-3B-Instruct
  uv run python scripts/check_vllm.py --image E:\\path\\to\\photo.jpg
"""

import sys

try:  # Korean output on cp949 consoles; guard so a quirky stdout can't crash the diagnostic
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from imgsearch.config import AppConfig
from imgsearch.core.models import FolderHit
from imgsearch.paths import app_paths


def _arg(name: str, default: str) -> str:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> int:
    cfg = AppConfig.load(app_paths().config_file)
    base_url = _arg("--base-url", cfg.chat_base_url)
    model = _arg("--model", cfg.chat_model)
    image = _arg("--image", "")

    print(f"vLLM 엔드포인트: {base_url}\n모델: {model}\n")
    ok = True

    # connectivity + chat backend
    try:
        from imgsearch.backends.openai_compat import OpenAICompatChatLLM

        chat = OpenAICompatChatLLM(base_url=base_url, model=model, api_key=cfg.chat_api_key)
        print("PASS  연결 (/models 응답)")
    except Exception as e:
        print(f"FAIL  연결 실패: {e}")
        print("\n→ vLLM 서버가 떠 있는지, base-url/model이 맞는지 확인하세요.")
        return 1

    q = "밤에 오토바이가 주차되어 있는 이미지"
    try:
        refined = chat.refine_query(q)
        print(f"PASS  refine_query → {refined!r}")
    except Exception as e:
        ok = False
        print(f"FAIL  refine_query: {e}")

    try:
        hits = [
            FolderHit("/f/a", "/f/a/x.jpg", "어두운 야간 오토바이 주차 장면", 0.42),
            FolderHit("/f/b", "/f/b/y.jpg", "야간 거리 사람", 0.31),
        ]
        summary = chat.summarize(q, hits)
        print(f"PASS  summarize → {summary[:80]!r}")
    except Exception as e:
        ok = False
        print(f"FAIL  summarize: {e}")

    # agentic planner JSON
    try:
        plan = chat.plan("사람과 오토바이가 함께 있는 이미지", {"0": "사람", "1": "오토바이"})
        if isinstance(plan, dict) and plan:
            print(f"PASS  plan → {plan}")
        else:
            print("WARN  plan이 빈 dict를 반환했습니다 (모델이 JSON을 안 지킴 → 규칙기반 폴백 사용)")
    except Exception as e:
        ok = False
        print(f"FAIL  plan: {e}")

    if image:
        try:
            from imgsearch.backends.openai_compat import OpenAICompatVLMCaptioner

            cap = OpenAICompatVLMCaptioner(base_url=cfg.vlm_base_url, model=cfg.vlm_model, api_key=cfg.vlm_api_key)
            text = cap.caption(image)
            print(f"PASS  caption({image}) → {text[:80]!r}")
        except Exception as e:
            ok = False
            print(f"FAIL  caption: {e}")

    print("\n" + ("전체 PASS — 실모드로 전환해도 됩니다." if ok else "일부 단계 실패 — 위 로그를 확인하세요."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
