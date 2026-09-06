from pathlib import Path

import pytest

from lab.storage import R2Store, r2_enabled


class _ResponseLessError(Exception):
    """Not every boto/network exception is a botocore ClientError with a dict `.response` --
    some (and any non-AWS S3-compatible error path, e.g. Cloudflare R2) may set the attribute to
    None outright, or omit "Error" from it. `getattr(e, "response", {})` only falls back to `{}`
    when the attribute is *missing*, not when it's falsy -- so a naive `.get()` chain on it
    crashes with AttributeError instead of reaching the `raise` below (2026-09-05 `queue list`
    incident)."""

    response = None


class _ErrorKeyIsNoneError(Exception):
    def __init__(self) -> None:
        super().__init__("weird error shape")
        self.response = {"Error": None}


def test_get_text_reraises_a_response_less_error_instead_of_crashing():
    class Client:
        def get_object(self, Bucket: str, Key: str) -> dict:
            raise _ResponseLessError("transient failure")

    store = R2Store("https://example.test", "bucket", client=Client())
    with pytest.raises(_ResponseLessError):
        store.get_text("some/key")


def test_get_text_reraises_when_error_key_is_none_instead_of_crashing():
    class Client:
        def get_object(self, Bucket: str, Key: str) -> dict:
            raise _ErrorKeyIsNoneError()

    store = R2Store("https://example.test", "bucket", client=Client())
    with pytest.raises(_ErrorKeyIsNoneError):
        store.get_text("some/key")


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
