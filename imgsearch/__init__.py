"""imgsearch — search an image dataset by natural-language (Korean) queries.

A local PySide6 desktop app: VLM-captioned, CLIP-embedded representative images
stored in ChromaDB, queried in natural language with optional LLM RAG summary.
Runs fully in a deterministic MOCK mode with zero ML dependencies; real backends
(jina-clip-v2 embedder, vLLM-served Qwen2.5-VL) activate from the Settings panel.
"""

__version__ = "0.1.0"
