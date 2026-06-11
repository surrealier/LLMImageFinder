"""Pick ONE representative image per leaf folder.

Default (an embedder is available): the member closest to the folder's mean embedding
— the most "typical" frame in the same space used for retrieval. Cheap fallback (no
embedder): the dHash medoid (most central frame), or the middle file if decoding fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from imgsearch.backends.base import Embedder
from imgsearch.core.models import RepImage


def _evenly_spaced(paths: Sequence[str], cap: int) -> list[str]:
    n = len(paths)
    if n <= cap:
        return list(paths)
    idx = np.linspace(0, n - 1, cap).round().astype(int)
    return [paths[i] for i in sorted(set(int(x) for x in idx))]


def _dhash_bits(path: str, hash_size: int = 8) -> Optional[np.ndarray]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            g = np.asarray(
                im.convert("L").resize((hash_size + 1, hash_size)), dtype=np.int16
            )
        return (g[:, 1:] > g[:, :-1]).reshape(-1)
    except Exception:
        return None


def _centroid_pick(sample: list[str], embedder: Embedder) -> Optional[int]:
    embs = embedder.embed_image(sample)
    if embs.shape[0] != len(sample) or embs.shape[0] == 0:
        return None
    mean = embs.mean(axis=0)
    n = float(np.linalg.norm(mean))
    if n > 0:
        mean = mean / n
    sims = embs @ mean
    return int(np.argmax(sims))


def _dhash_medoid(sample: list[str]) -> Optional[int]:
    bits = [_dhash_bits(p) for p in sample]
    valid = [(i, b) for i, b in enumerate(bits) if b is not None]
    if not valid:
        return None
    idxs = [i for i, _ in valid]
    mats = np.stack([b for _, b in valid])  # (m, 64) bool
    # pairwise Hamming distances, pick the row with the smallest total distance
    dists = np.zeros((len(idxs), len(idxs)), dtype=np.int32)
    for a in range(len(idxs)):
        dists[a] = np.count_nonzero(mats[a] ^ mats, axis=1)
    medoid_local = int(np.argmin(dists.sum(axis=1)))
    return idxs[medoid_local]


def choose_representative(
    image_paths: Sequence[str],
    embedder: Optional[Embedder] = None,
    max_members: int = 64,
) -> RepImage:
    paths = list(image_paths)
    n = len(paths)
    if n == 0:
        raise ValueError("choose_representative: empty image list")
    folder = str(Path(paths[0]).parent)
    if n == 1:
        return RepImage(folder, paths[0], 1, "only")

    sample = _evenly_spaced(paths, max_members)

    if embedder is not None:
        pick = _centroid_pick(sample, embedder)
        if pick is not None:
            return RepImage(folder, sample[pick], n, "centroid")

    pick = _dhash_medoid(sample)
    if pick is not None:
        return RepImage(folder, sample[pick], n, "dhash-medoid")

    return RepImage(folder, sample[len(sample) // 2], n, "middle")
