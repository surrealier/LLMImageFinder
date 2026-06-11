"""Verify all modules compile and the real-backend fallback to mock works."""

import glob
import py_compile

for f in glob.glob("imgsearch/**/*.py", recursive=True):
    py_compile.compile(f, doraise=True)
print("compile-all OK")

from imgsearch.config import AppConfig
from imgsearch.core.registry import build_backends

# vLLM + jina-clip requested but unavailable -> graceful mock fallback + degraded
cfg = AppConfig(
    embedder_backend="jina-clip",
    vlm_backend="vllm",
    chat_backend="vllm",
    vlm_base_url="http://localhost:8000/v1",
    chat_base_url="http://localhost:8000/v1",
)
emb, cap, chat = build_backends(cfg)
print("embedder:", emb.name, "| captioner:", getattr(cap, "name", None), "| chat:", chat.name)
print("degraded:", cfg.degraded)
print("reason:", cfg.degraded_reason[:140])
assert emb.name == "mock" and chat.name == "mock" and cfg.degraded, "fallback failed"
print("FALLBACK OK")

import imgsearch.backends.openai_compat as oc

print(
    "openai_compat OK:",
    hasattr(oc, "OpenAICompatVLMCaptioner"),
    hasattr(oc, "OpenAICompatChatLLM"),
)
