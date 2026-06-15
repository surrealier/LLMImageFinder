"""리프 폴더 하나당 "대표 이미지" 1장을 고른다.

선택 전략(우선순위 순):
  * 기본(임베더가 있을 때): 폴더 평균 임베딩에 가장 가까운 멤버 — 검색에 쓰는 바로
    그 임베딩 공간에서 가장 "전형적인" 프레임이다.
  * 저렴한 대체(임베더 없음): dHash 메도이드(가장 중앙에 있는 프레임).
  * 그래도 실패하면: 가운데 순번 파일을 그냥 사용.

대표 이미지는 폴더 단위 인덱싱에서 썸네일/임베딩의 기준이 되므로, 폴더를 잘 대변할수록
검색 품질이 좋아진다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from imgsearch.backends.base import Embedder
from imgsearch.core.models import RepImage


def _evenly_spaced(paths: Sequence[str], cap: int) -> list[str]:
    """경로 목록에서 ``cap`` 개 이하가 되도록 "고르게 분산된" 표본을 뽑는다.

    이미지가 수백~수천 장인 폴더에서 전부를 디코딩/임베딩하면 비용이 폭증하므로,
    앞쪽만 자르지 않고 전체 구간에 걸쳐 균등 표본을 취해 대표성을 유지한다.
    """
    n = len(paths)
    if n <= cap:
        # 이미 한도 이하면 그대로 복사 반환(표본 추출 불필요).
        return list(paths)
    # 0..n-1 구간을 cap 개의 등간격 지점으로 나눠 반올림한 정수 인덱스를 만든다.
    idx = np.linspace(0, n - 1, cap).round().astype(int)
    # 반올림으로 인덱스가 겹칠 수 있으므로 set 으로 중복 제거 후 정렬해 순서를 보존한다.
    return [paths[i] for i in sorted(set(int(x) for x in idx))]


def _dhash_bits(path: str, hash_size: int = 8) -> Optional[np.ndarray]:
    """이미지의 dHash(difference hash) 비트 벡터를 계산한다(실패 시 None).

    그레이스케일로 변환 후 (hash_size+1) x hash_size 로 축소하고, 가로 방향 인접
    픽셀의 밝기 대소 관계를 비교해 hash_size*hash_size(기본 64) 길이의 불리언 배열을
    만든다. 이는 "지각적 해시(perceptual hash)"로, 비슷해 보이는 이미지끼리 비트가
    비슷해져 해밍 거리로 시각적 유사도를 근사할 수 있다.
    """
    try:
        from PIL import Image  # PIL 은 무거운 선택적 의존성이라 함수 내부에서 늦게 import.

        with Image.open(path) as im:
            # "L"(8비트 그레이스케일)로 변환해 색상 영향을 제거하고 비교를 단순화한다.
            # 폭을 +1 더 크게 잡는 이유: 인접 열 비교(차분)에 한 칸이 더 필요하기 때문.
            g = np.asarray(
                im.convert("L").resize((hash_size + 1, hash_size)), dtype=np.int16
            )
        # 각 행에서 오른쪽 픽셀이 왼쪽보다 밝은지(>)를 비교해 1차원 비트 벡터로 펼친다.
        return (g[:, 1:] > g[:, :-1]).reshape(-1)
    except Exception:
        # 손상된 파일/디코딩 실패 등은 None 으로 돌려 호출부가 우아하게 대체 처리하게 한다.
        return None


def _centroid_pick(sample: list[str], embedder: Embedder) -> Optional[int]:
    """표본을 임베딩해 평균 벡터에 가장 가까운(코사인 최대) 멤버의 인덱스를 고른다.

    검색에 사용하는 동일 임베딩 공간에서 "평균적/전형적" 이미지를 대표로 삼는다.
    임베딩 개수가 입력 수와 다르거나 비면 신뢰할 수 없으므로 None 을 반환한다.
    """
    embs = embedder.embed_image(sample)
    if embs.shape[0] != len(sample) or embs.shape[0] == 0:
        return None
    mean = embs.mean(axis=0)
    # 평균 벡터를 단위 길이로 정규화한다. 임베딩이 이미 정규화돼 있다면 내적(embs @ mean)이
    # 곧 코사인 유사도가 되어, argmax 가 "평균에 가장 가까운" 멤버를 가리킨다.
    n = float(np.linalg.norm(mean))
    if n > 0:  # 0 나눗셈 방지(모든 벡터가 상쇄돼 평균이 영벡터인 극단 케이스 대비).
        mean = mean / n
    sims = embs @ mean
    return int(np.argmax(sims))


def _dhash_medoid(sample: list[str]) -> Optional[int]:
    """dHash 해밍 거리 기준 메도이드(다른 모든 멤버와 가장 가까운 멤버) 인덱스를 고른다.

    임베더가 없을 때의 대체 전략. 디코딩 가능한 이미지가 하나도 없으면 None.
    """
    bits = [_dhash_bits(p) for p in sample]
    # 디코딩에 성공한(해시가 있는) 항목만 원래 인덱스와 함께 추린다.
    valid = [(i, b) for i, b in enumerate(bits) if b is not None]
    if not valid:
        return None
    idxs = [i for i, _ in valid]  # 유효 항목의 원본(sample 기준) 인덱스
    mats = np.stack([b for _, b in valid])  # (m, 64) 불리언 행렬: m개 이미지의 해시
    # 모든 쌍의 해밍 거리 행렬을 채운 뒤, 행 합이 가장 작은(=다른 모두와 가장 가까운) 행을 고른다.
    dists = np.zeros((len(idxs), len(idxs)), dtype=np.int32)
    for a in range(len(idxs)):
        # XOR 로 비트 차이를 만들고 1의 개수를 세면 한 이미지 vs 전체의 해밍 거리(브로드캐스팅).
        dists[a] = np.count_nonzero(mats[a] ^ mats, axis=1)
    medoid_local = int(np.argmin(dists.sum(axis=1)))  # valid 내부 기준 인덱스
    return idxs[medoid_local]  # 원본 sample 기준 인덱스로 환원해 반환.


def choose_representative(
    image_paths: Sequence[str],
    embedder: Optional[Embedder] = None,
    max_members: int = 64,
) -> RepImage:
    """폴더의 이미지 목록에서 대표 이미지 1장을 골라 :class:`RepImage` 로 반환한다.

    선택 경로(앞쪽이 우선): centroid(임베더 평균 최근접) → dhash-medoid → middle.
    반환되는 RepImage 의 마지막 인자는 어떤 방법이 쓰였는지를 기록하는 라벨이며,
    member_count 에는 (표본이 아닌) 폴더의 전체 이미지 수 ``n`` 을 담는다.
    """
    paths = list(image_paths)
    n = len(paths)
    if n == 0:
        # 빈 폴더는 호출부 계약 위반이므로 명시적으로 실패시킨다.
        raise ValueError("choose_representative: empty image list")
    folder = str(Path(paths[0]).parent)
    if n == 1:
        # 이미지가 1장뿐이면 선택의 여지가 없다("only" 라벨).
        return RepImage(folder, paths[0], 1, "only")

    # 비용 제한을 위해 균등 표본만 가지고 대표를 고른다(아래 모든 전략이 이 표본을 사용).
    sample = _evenly_spaced(paths, max_members)

    # 1순위: 임베더가 있으면 임베딩 평균에 가장 가까운 전형적 프레임.
    if embedder is not None:
        pick = _centroid_pick(sample, embedder)
        if pick is not None:
            return RepImage(folder, sample[pick], n, "centroid")

    # 2순위: 임베더가 없거나 임베딩이 실패하면 dHash 메도이드로 대체.
    pick = _dhash_medoid(sample)
    if pick is not None:
        return RepImage(folder, sample[pick], n, "dhash-medoid")

    # 최후의 보루: 모든 디코딩이 실패해도 무언가는 반환해야 하므로 가운데 파일을 쓴다.
    return RepImage(folder, sample[len(sample) // 2], n, "middle")
