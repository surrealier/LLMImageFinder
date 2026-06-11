from imgsearch.config import DEFAULT_IMAGE_EXTS
from imgsearch.index import walker


def test_iter_leaf_folders_count(sample_ds, n_scenes):
    folders = dict(walker.iter_leaf_folders(str(sample_ds), DEFAULT_IMAGE_EXTS))
    assert len(folders) == n_scenes
    for _d, imgs in folders.items():
        assert len(imgs) == 3
        assert all(p.lower().endswith((".jpg", ".jpeg", ".png")) for p in imgs)


def test_list_images_sorted(sample_ds):
    folders = list(walker.iter_leaf_folders(str(sample_ds), DEFAULT_IMAGE_EXTS))
    _d, imgs = folders[0]
    assert imgs == sorted(imgs, key=lambda p: p.lower())


def test_empty_dir(tmp_path):
    assert list(walker.iter_leaf_folders(str(tmp_path), DEFAULT_IMAGE_EXTS)) == []
