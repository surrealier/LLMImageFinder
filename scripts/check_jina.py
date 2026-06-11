"""Verify CUDA + that jina-clip-v2 loads and embeds (text & image) into 512-d cosine space."""

import sys

import numpy as np
import torch

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

from imgsearch.backends.jina_clip import JinaClipEmbedder

emb = JinaClipEmbedder(device="auto", dim=512)
print("device:", emb.device, "dim:", emb.dim)

t = emb.embed_text(["밤에 오토바이가 주차되어 있는 거리", "fire and smoke"])
print("text emb:", t.shape, "norm0:", round(float(np.linalg.norm(t[0])), 4))

img = sys.argv[1] if len(sys.argv) > 1 else None
if img:
    v = emb.embed_image([img])
    print("image emb:", v.shape, "norm:", round(float(np.linalg.norm(v[0])), 4))
    print("cos(text0,img):", round(float(t[0] @ v[0]), 4))
print("JINA OK")
