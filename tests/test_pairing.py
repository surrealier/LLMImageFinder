from imgsearch.config import DEFAULT_IMAGE_EXTS
from imgsearch.index import pairing, walker


def _first_folder(sample_ds):
    return list(walker.iter_leaf_folders(str(sample_ds), DEFAULT_IMAGE_EXTS))[0]


def test_sidecar_for(sample_ds):
    _d, imgs = _first_folder(sample_ds)
    txt = pairing.sidecar_for(imgs[0])
    assert txt and "frame" in txt


def test_folder_sidecar_text(sample_ds):
    d, imgs = _first_folder(sample_ds)
    ftxt = pairing.folder_sidecar_text(d, imgs)
    assert len(ftxt) > 0
    assert len(ftxt) <= 800


def test_sidecar_missing(tmp_path):
    img = tmp_path / "lonely.jpg"
    img.write_bytes(b"\xff\xd8\xff")  # not a real jpeg, just a path
    assert pairing.sidecar_for(str(img)) == ""
