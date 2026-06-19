from imgsearch import koutil
from imgsearch.backends.mock import MockChatLLM
from imgsearch.core.models import FolderHit


def test_expand_query_ko_to_en():
    out = koutil.expand_query("밤에 오토바이가 주차되어 있는 이미지")
    assert "night" in out
    assert "motorcycle" in out


def test_tokenize_drops_meta_words():
    toks = koutil.tokenize("불과 연기가 있는 이미지")
    assert "이미지" not in toks
    assert any("불" in t for t in toks)


def test_strip_one_josa_keeps_noun_via_both_forms():
    # original token preserved even if it ends in a josa char
    feats = dict(koutil.feature_tokens("오토바이"))
    assert "w:오토바이" in feats


def test_concept_hints():
    hints = koutil.concept_hints("fire and smoke in a building")
    assert "fire" in hints and "smoke" in hints


def test_mock_refine_and_summary():
    chat = MockChatLLM()
    r = chat.refine_query("불과 연기가 있는 이미지")  # KO query still expands via KO2EN
    assert "fire" in r and "smoke" in r
    assert "No images matched" in chat.summarize("query", [])
    s = chat.summarize("query", [FolderHit("a/b/fire_smoke", "x.jpg", "caption", 0.9, 3)])
    assert "fire_smoke" in s
