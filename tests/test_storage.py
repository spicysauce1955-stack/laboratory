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


class _ErrorKeyIsBareStringError(Exception):
    """Some non-AWS S3-compatible provider (or a genuinely malformed response) may put a bare
    string under "Error" instead of the AWS-shaped {"Code": ..., "Message": ...} dict -- a naive
    `.get("Code")` on that string crashes with AttributeError instead of reaching the `raise`
    below (the same crash class the None-response and None-Error fixes closed, one layer
    deeper)."""

    def __init__(self) -> None:
        super().__init__("weird error shape")
        self.response = {"Error": "AccessDenied"}


class _ResponseIsBareStringError(Exception):
    """A third layer of the same crash class: `.response` can be present-but-truthy while not
    being a dict at all (a bare string), which `getattr(e, "response", None) or {}` doesn't catch
    -- `or {}` only kicks in when `.response` is falsy/missing, so a non-dict-but-truthy
    `.response` still reaches `.get("Error")` and crashes with AttributeError instead of the
    `raise` below (2026-09-06 review, one layer deeper than the Error-is-bare-string fix)."""

    def __init__(self) -> None:
        super().__init__("weird error shape")
        self.response = "not a dict at all"


def test_get_text_reraises_when_error_is_a_bare_string_instead_of_crashing():
    class Client:
        def get_object(self, Bucket: str, Key: str) -> dict:
            raise _ErrorKeyIsBareStringError()

    store = R2Store("https://example.test", "bucket", client=Client())
    with pytest.raises(_ErrorKeyIsBareStringError):
        store.get_text("some/key")


def test_get_text_reraises_when_response_is_a_bare_string_instead_of_crashing():
    class Client:
        def get_object(self, Bucket: str, Key: str) -> dict:
            raise _ResponseIsBareStringError()

    store = R2Store("https://example.test", "bucket", client=Client())
    with pytest.raises(_ResponseIsBareStringError):
        store.get_text("some/key")


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


class _ClientError(Exception):
    """botocore's ClientError shape without importing botocore: `.response["Error"]["Code"]`."""

    def __init__(self, code: str) -> None:
        super().__init__(f"An error occurred ({code}) when calling the HeadObject operation")
        self.response = {"Error": {"Code": code}}


def _head_raising(exc: Exception) -> R2Store:
    class Client:
        def head_object(self, Bucket: str, Key: str) -> dict:
            raise exc

    return R2Store("https://example.test", "bucket", client=Client())


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_exists_is_false_only_for_a_real_not_found(code: str):
    assert _head_raising(_ClientError(code)).exists("queue/cancelled/reg-a") is False


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("Could not connect to the endpoint URL"),  # a network blip
        _ClientError("403"),  # HeadObject permission missing on the token
        _ClientError("AccessDenied"),
        _ClientError("SlowDown"),
        _ResponseLessError("transient failure"),
        _ResponseIsBareStringError(),
    ],
)
def test_exists_reraises_anything_that_is_not_a_not_found(exc: Exception):
    """`exists()` answers the scheduler's race-critical pre-submit cancel/hold re-checks. "I could
    not ask" read as "no marker" made both fail OPEN: a cancel that landed during a network blip,
    or on a token without HeadObject, rented the machine anyway."""
    with pytest.raises(type(exc)):
        _head_raising(exc).exists("queue/cancelled/reg-a")


def test_exists_is_true_when_head_succeeds():
    class Client:
        def head_object(self, Bucket: str, Key: str) -> dict:
            return {}

    assert R2Store("https://example.test", "bucket", client=Client()).exists("k") is True
