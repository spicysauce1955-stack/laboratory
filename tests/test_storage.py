from pathlib import Path

import pytest

from lab.storage import R2Store, r2_enabled


@pytest.mark.skipif(not r2_enabled(), reason="R2 not configured (set LAB_R2_ENDPOINT + creds)")
def test_r2_round_trip(tmp_path: Path):
    store = R2Store.from_env()
    assert store is not None
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("hello")
    (src / "sub" / "b.txt").write_text("world")

    prefix = "test/pytest-roundtrip"
    assert store.upload_dir(src, prefix) == 2

    dst = tmp_path / "dst"
    assert store.download_dir(prefix, dst) == 2
    assert (dst / "a.txt").read_text() == "hello"
    assert (dst / "sub" / "b.txt").read_text() == "world"

    for key in (f"{prefix}/a.txt", f"{prefix}/sub/b.txt"):
        store._s3.delete_object(Bucket=store.bucket, Key=key)
