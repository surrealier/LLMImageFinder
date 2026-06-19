"""데모와 테스트용으로 작은 합성 이미지-텍스트 데이터셋을 생성한다.

중첩된 폴더 트리를 만든다: 각 말단(leaf) 폴더는 한 장면의 *거의 동일한* 프레임 여러 장과
이미지별 한국어 사이드카 ``.txt`` 캡션을 담는다. 폴더명은 영어 개념 단어를 쓰는데, 이는
결정론적 mock 백엔드가 (:mod:`imgsearch.koutil`의 KO->EN 사전을 통해) 한국어 질의와
매칭할 수 있게 하기 위함이다.
"""

from __future__ import annotations

import random
from pathlib import Path

# (카테고리, 장면 폴더명, 그리기 키, 한국어 캡션) 튜플 목록.
# 그리기 키(builder_key)는 _draw_scene가 어떤 도형을 그릴지 결정하는 식별자다.
_SCENES = [
    ("outdoor_night", "motorcycle_parked_night", "motorcycle_night",
     "A motorcycle parked by the road at night"),
    ("outdoor_night", "street_people_night", "people_night",
     "People walking on a street at night"),
    ("street_day", "people_crosswalk_day", "people_day",
     "People crossing a crosswalk on a daytime avenue"),
    ("street_day", "cars_traffic_day", "cars_day",
     "Cars driving on a road during the day"),
    ("emergency", "fire_smoke_building", "fire",
     "A fire with flames and smoke at a building"),
    ("emergency", "smoke_only_field", "smoke",
     "Smoke spreading across a field"),
    ("parking", "cars_parking_lot_day", "cars_day",
     "Cars parked in a daytime parking lot"),
    ("nature", "beach_sea_day", "beach",
     "A clear daytime beach by the sea"),
]


def _draw_scene(key: str, w: int, h: int, jitter: random.Random):
    """그리기 키에 맞춰 간단한 도형으로 한 장의 합성 장면 이미지를 그려 반환한다.

    jitter(시드된 난수)를 써서 같은 장면이라도 프레임마다 위치를 약간씩 흔들어
    '거의 동일한' 프레임 묶음을 만든다(폴더 단위 중복 장면을 흉내).
    """
    from PIL import Image, ImageDraw

    night = key.endswith("_night") or key in {"fire", "smoke"} and False
    # 배경색: 야간 장면은 어둡게, 해변은 하늘빛, 그 외는 회색 톤으로.
    if key.endswith("_night"):
        bg = (14, 18, 32)
    elif key == "beach":
        bg = (120, 190, 225)
    else:
        bg = (175, 198, 214)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    # 프레임마다 전체 객체를 (ox, oy)만큼 흔들어 미세한 차이를 준다.
    ox, oy = jitter.randint(-8, 8), jitter.randint(-6, 6)

    def rect(x0, y0, x1, y1, fill):
        """jitter 오프셋(ox, oy)을 적용해 사각형을 그리는 헬퍼."""
        d.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=fill)

    def ell(x0, y0, x1, y1, fill):
        """jitter 오프셋(ox, oy)을 적용해 타원을 그리는 헬퍼."""
        d.ellipse([x0 + ox, y0 + oy, x1 + ox, y1 + oy], fill=fill)

    # 바닥/도로: 해변이 아니면 야간은 더 어두운 회색, 주간은 밝은 회색 노면을 깐다.
    if key != "beach":
        rect(0, h - 70, w, h, (45, 45, 52) if key.endswith("_night") else (95, 95, 105))
    else:
        # 해변은 모래색 바닥.
        rect(0, h - 70, w, h, (210, 200, 150))

    # 키 안에 포함된 개념 단어에 따라 해당 객체들을 겹쳐 그린다(복수 동시 적용 가능).
    if "motorcycle" in key:
        rect(110, 120, 210, 150, (30, 30, 38))           # 차체
        ell(108, 150, 148, 190, (10, 10, 10))            # 바퀴
        ell(178, 150, 218, 190, (10, 10, 10))            # 바퀴
        d.line([130 + ox, 122 + oy, 120 + ox, 100 + oy], fill=(60, 60, 70), width=4)
    if "people" in key:
        # 사람 수도 jitter로 3~5명 사이에서 변동시켜 프레임마다 다르게 보이게 한다.
        for i in range(jitter.randint(3, 5)):
            x = 40 + i * 55 + jitter.randint(-6, 6)
            ell(x, 150, x + 18, 200, (40, 60, 110))      # 몸통
            ell(x + 2, 132, x + 16, 150, (225, 200, 170))  # 머리
    if "cars" in key:
        for i in range(2):
            x = 60 + i * 130
            # 첫 차는 빨강, 둘째 차는 파랑으로 구분.
            rect(x, 140, x + 90, 180, (150, 40, 40) if i == 0 else (40, 70, 150))
            rect(x + 12, 122, x + 78, 142, (190, 210, 230))  # 창문
    if key == "fire":
        rect(120, 90, 200, 180, (70, 60, 55))            # 건물
        # 들쭉날쭉한 다각형으로 불꽃 모양을 표현(주황색).
        d.polygon([(150 + ox, 170 + oy), (130 + ox, 110 + oy), (160 + ox, 130 + oy),
                   (150 + ox, 80 + oy), (175 + ox, 125 + oy), (180 + ox, 165 + oy)],
                  fill=(235, 120, 20))
        ell(140, 60, 210, 110, (120, 120, 125))          # 연기
    if key == "smoke":
        # 회색 타원 여러 개를 흩뿌려 자욱한 연기 느낌을 낸다.
        for i in range(4):
            ell(70 + i * 45, 70 + jitter.randint(-10, 10), 150 + i * 45, 150,
                (140, 140, 145))
    if key == "beach":
        ell(230, 30, 290, 90, (250, 240, 120))           # 태양
    return img


def generate_sample_dataset(
    root: str | Path, per_scene: int = 5, seed: int = 7, size=(320, 240)
) -> Path:
    """``root`` 아래에 합성 데이터셋을 만들고 그 경로를 반환한다.

    per_scene: 장면(폴더)당 생성할 프레임 수. seed: 재현 가능한 jitter를 위한 시드.
    """
    root = Path(root)
    w, h = size
    for category, scene, key, caption in _SCENES:
        leaf = root / category / scene
        leaf.mkdir(parents=True, exist_ok=True)
        # 장면명을 시드에 섞어, 같은 seed라도 장면마다 다른(그러나 재현 가능한) 난수열을 쓴다.
        rng = random.Random(f"{seed}:{scene}")
        for i in range(per_scene):
            img = _draw_scene(key, w, h, rng)
            # 파일명 stem에 3자리 0패딩 인덱스를 붙여 정렬과 식별을 쉽게 한다.
            stem = f"{scene}_{i:03d}"
            img.save(leaf / f"{stem}.jpg", quality=88)
            # 이미지 옆에 같은 이름의 사이드카 .txt 캡션을 둔다(프레임 번호 포함).
            (leaf / f"{stem}.txt").write_text(
                f"{caption} (frame {i + 1}/{per_scene})", encoding="utf-8"
            )
    return root


if __name__ == "__main__":  # pragma: no cover
    # 스크립트로 직접 실행 시: 첫 인자를 출력 경로로 쓰고, 없으면 기본 폴더명을 사용.
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"
    p = generate_sample_dataset(out)
    print(f"Sample dataset created at: {p.resolve()}")
