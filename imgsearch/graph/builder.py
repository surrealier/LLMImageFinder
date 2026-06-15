"""데이터셋의 YOLO 라벨을 훑어 GraphRecord들을 만든다.

그래프의 "진실의 원천(source of truth)"은 디스크에 있는 데이터셋 라벨이지(크로마에 저장된
문서가 아니라), 그래서 그래프는 실제 어노테이션을 그대로 반영한다. 클래스 id는 캡션과
"똑같이" ``cfg.class_names``를 통해 표시 이름으로 변환한다 — 명명 규칙을 하나로 통일하기 위함.
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

from imgsearch.config import AppConfig
from imgsearch.graph.base import GraphRecord
from imgsearch.index import labels, walker

# 진행률 콜백 타입 별칭: (처리한 개수, 현재 상태 메시지)를 받는 호출 가능 객체.
ProgressCb = Callable[[int, str], None]


def iter_graph_records(root: str, cfg: AppConfig) -> Iterable[GraphRecord]:
    """YOLO 라벨 박스가 있는 이미지마다 GraphRecord를 하나씩 생성(yield)한다.

    제너레이터라서 전체를 메모리에 모으지 않고 스트리밍 처리할 수 있다(대용량 데이터셋 대비).
    """
    # root가 비었거나 실제 디렉터리가 아니면 아무것도 내지 않고 조용히 종료.
    if not root or not os.path.isdir(root):
        return
    # 리프 폴더 단위로 순회하며 각 폴더의 이미지 목록을 받는다.
    for dirpath, images in walker.iter_leaf_folders(root, cfg.image_exts):
        for img in images:
            # 이미지에 대응하는 라벨 파일 경로를 찾는다(없으면 None).
            lp = labels.label_path_for(img)
            if lp is None:
                continue  # 라벨이 없는 이미지는 그래프 대상이 아니다.
            boxes = labels.parse_yolo(lp)
            if not boxes:
                continue  # 라벨 파일은 있으나 박스가 비어 있으면 건너뛴다.
            # 박스의 class id들을 cfg.class_names로 표시 이름 목록으로 변환(캡션과 동일 규칙).
            names = labels.names_in(boxes, cfg.class_names)
            if names:
                yield GraphRecord(image_path=img, folder=dirpath, classes=names)


def build_records(root: str, cfg: AppConfig) -> list[GraphRecord]:
    """iter_graph_records의 결과를 모두 모아 리스트로 반환하는 편의 함수.

    제너레이터를 한 번에 소비해야 할 때(예: 길이를 알아야 하거나 그래프 build에 통째로
    넘길 때) 사용한다.
    """
    return list(iter_graph_records(root, cfg))
