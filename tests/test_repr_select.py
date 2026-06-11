from imgsearch.backends.mock import MockEmbedder
from imgsearch.config import DEFAULT_IMAGE_EXTS
from imgsearch.index import walker
from imgsearch.index.repr_select import choose_representative


def _first_folder(sample_ds):
    return list(walker.iter_leaf_folders(str(sample_ds), DEFAULT_IMAGE_EXTS))[0]


def test_centroid_with_embedder(sample_ds):
    _d, imgs = _first_folder(sample_ds)
    rep = choose_representative(imgs, MockEmbedder(512), max_members=64)
    assert rep.image_path in imgs
    assert rep.method == "centroid"
    assert rep.member_count == len(imgs)


def test_fallback_no_embedder(sample_ds):
    _d, imgs = _first_folder(sample_ds)
    rep = choose_representative(imgs, None, max_members=64)
    assert rep.image_path in imgs
    assert rep.method in ("dhash-medoid", "middle")


def test_single_image():
    rep = choose_representative(["/nonexistent/only.jpg"])
    assert rep.method == "only"
    assert rep.member_count == 1
