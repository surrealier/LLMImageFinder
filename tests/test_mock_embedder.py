import numpy as np

from imgsearch.backends.mock import MockEmbedder
from imgsearch.config import DEFAULT_IMAGE_EXTS
from imgsearch.index import walker


def test_text_determinism_dim_norm():
    e = MockEmbedder(512)
    a = e.embed_text(["밤에 오토바이가 주차되어 있는 거리"])
    b = e.embed_text(["밤에 오토바이가 주차되어 있는 거리"])
    assert a.shape == (1, 512)
    np.testing.assert_allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-5


def test_image_dim_norm(sample_ds):
    e = MockEmbedder(512)
    _d, imgs = list(walker.iter_leaf_folders(str(sample_ds), DEFAULT_IMAGE_EXTS))[0]
    v = e.embed_image(imgs[:1])
    assert v.shape == (1, 512)
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5


def test_empty_inputs():
    e = MockEmbedder(512)
    assert e.embed_text([]).shape == (0, 512)
    assert e.embed_image([]).shape == (0, 512)
