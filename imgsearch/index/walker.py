"""데이터셋 루트를 순회하며 리프(leaf) 폴더(이미지를 담고 있는 디렉터리)를 하나씩 산출한다.

리프 폴더란 "이미지 파일을 직접 담고 있는 디렉터리"를 뜻한다. 인덱싱 파이프라인은
이 모듈이 산출하는 (폴더 경로, 이미지 목록) 쌍을 작업 단위로 삼는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sequence

# 순회 중 건너뛸 디렉터리 이름들(소문자 비교). 버전 관리/캐시/썸네일/휴지통 등은
# 데이터셋 콘텐츠가 아니므로 탐색 비용과 오탐을 줄이기 위해 미리 제외한다.
_SKIP_DIR_NAMES = {".git", "__pycache__", "thumbs", ".thumbs", ".cache", "$recycle.bin"}


def _is_image(name: str, exts: Sequence[str]) -> bool:
    """파일 이름의 확장자가 허용된 이미지 확장자 목록(``exts``)에 포함되는지 검사한다."""
    # 확장자만 떼어내 소문자로 정규화한다(.JPG 와 .jpg 를 동일하게 취급하기 위함).
    ext = os.path.splitext(name)[1].lower()
    return ext in exts


def list_images(dirpath: str | Path, exts: Sequence[str]) -> list[str]:
    """``dirpath`` 바로 안에 있는 이미지 파일들의 절대 경로를 정렬해서 반환한다.

    하위 디렉터리는 재귀하지 않고 해당 폴더 직속 파일만 본다(리프 폴더 단위 처리를 위함).
    """
    d = Path(dirpath)
    try:
        # os.scandir 는 os.listdir 보다 빠르다(파일 타입 정보를 함께 얻어 stat 호출을 줄임).
        entries = os.scandir(d)
    except OSError:
        # 권한 없음/경로 사라짐 등으로 열 수 없으면 빈 목록으로 조용히 넘어간다.
        return []
    out: list[str] = []
    # scandir 의 결과는 OS 핸들을 점유하므로 with 로 확실히 닫는다.
    with entries:
        for e in entries:
            try:
                # 디렉터리/심볼릭링크가 아닌 실제 파일이면서 이미지 확장자인 것만 채택.
                if e.is_file() and _is_image(e.name, exts):
                    # Path 로 한 번 감싸 OS별 경로 구분자를 정규화한 뒤 문자열로 저장.
                    out.append(str(Path(e.path)))
            except OSError:
                # 개별 항목 stat 실패(끊긴 링크 등)는 그 항목만 건너뛰고 계속 진행.
                continue
    # 대소문자 무시 정렬로 OS/파일시스템과 무관하게 결정적(deterministic)인 순서를 보장.
    out.sort(key=lambda p: p.lower())
    return out


def iter_leaf_folders(
    root: str | Path, exts: Sequence[str]
) -> Iterator[tuple[str, list[str]]]:
    """이미지를 1장 이상 가진 모든 디렉터리에 대해 ``(dirpath, image_paths)`` 를 산출한다.

    제너레이터이므로 전체 트리를 메모리에 한꺼번에 올리지 않고 폴더 단위로 흘려보낸다
    (대용량 데이터셋에서도 메모리 사용량이 일정하게 유지된다).
    """
    root = str(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        # 숨김 폴더(.으로 시작)와 캐시/제외 대상 폴더를 그 자리에서 제거한다.
        # os.walk 는 dirnames 리스트를 "제자리 수정"하면 그 하위로 더 내려가지 않으므로,
        # 새 리스트를 만들어 슬라이스 대입(dirnames[:] = ...)으로 가지치기(pruning)한다.
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d.lower() not in _SKIP_DIR_NAMES
        ]
        images = list_images(dirpath, exts)
        # 이미지가 하나라도 있는 폴더만 작업 단위로 인정한다(중간 경로 폴더는 건너뜀).
        if images:
            yield str(Path(dirpath)), images
