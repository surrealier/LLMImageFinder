from imgsearch.index import labels


def test_is_yolo_text():
    assert labels.is_yolo_text("3 0.355 0.596 0.710 0.806\n1 0.5 0.4 0.4 0.8")
    assert not labels.is_yolo_text("밤에 오토바이가 주차된 거리입니다")
    assert not labels.is_yolo_text("")


def test_parse_yolo(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("3 0.3 0.5 0.7 0.8\n3 0.1 0.1 0.1 0.1\n1 0.5 0.4 0.4 0.8\n")
    boxes = labels.parse_yolo(p)
    assert len(boxes) == 3
    assert labels.class_histogram(boxes) == {3: 2, 1: 1}


def test_yolo_caption_with_and_without_names(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("3 0.3 0.5 0.7 0.8\n3 0.1 0.1 0.1 0.1\n1 0.5 0.4 0.4 0.8\n")
    cap = labels.yolo_caption(p, {"3": "불", "1": "오토바이"})
    assert "불×2" in cap and "오토바이" in cap
    cap2 = labels.yolo_caption(p, None)
    assert "#3×2" in cap2


def test_caption_merges_ids_with_same_name():
    # ids 0,1,3 -> "휴대폰", 4 -> "휴대폰 후면 카메라": names must aggregate, not repeat
    boxes = [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (3, 0, 0, 0, 0), (4, 0, 0, 0, 0)]
    names = {"0": "휴대폰", "1": "휴대폰", "3": "휴대폰", "4": "휴대폰 후면 카메라"}
    cap = labels.caption_from_boxes(boxes, names)
    assert "휴대폰×3" in cap
    assert "휴대폰 후면 카메라" in cap
    assert cap.count("휴대폰×") == 1  # not "휴대폰×2, 휴대폰..." fragments


def test_yolo_caption_non_label(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("이것은 실제 캡션 문장입니다")
    assert labels.yolo_caption(p) == ""  # not YOLO -> empty
