import json

from imgsearch.config import AppConfig


def test_round_trip_non_default_fields(tmp_path):
    p = tmp_path / "config.json"
    cfg = AppConfig(
        dataset_root="D:/데이터/학습",
        index_granularity="image",
        class_names={"0": "사람", "4": "휴대폰 후면 카메라"},
        top_k=7,
        score_threshold=0.25,
        image_exts=[".jpg"],
        embedder_backend="jina-clip",
    )
    cfg.save(p)
    loaded = AppConfig.load(p)
    assert loaded == cfg  # runtime-only fields excluded via compare=False


def test_corrupt_json_returns_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{broken", encoding="utf-8")
    assert AppConfig.load(p) == AppConfig()


def test_unknown_and_runtime_keys_ignored_on_load(tmp_path):
    p = tmp_path / "config.json"
    raw = AppConfig(top_k=9).to_dict()
    raw["future_key"] = 123  # forward compat: a newer version's key
    raw["degraded"] = True  # runtime-only: must never be accepted from disk
    raw["degraded_reason"] = "x"
    p.write_text(json.dumps(raw), encoding="utf-8")
    loaded = AppConfig.load(p)
    assert loaded.top_k == 9
    assert loaded.degraded is False
    assert loaded.degraded_reason == ""


def test_degraded_state_never_persisted(tmp_path):
    p = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.mark_degraded("백엔드 없음")
    cfg.save(p)
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert "degraded" not in on_disk
    assert "degraded_reason" not in on_disk
