"""A job's output dir is rsynced from a rented box we do not control, so it is untrusted input.

``rsync -a`` reproduces symlinks verbatim. A hostile Vast host that plants
``results.csv -> ~/.ssh/sky-key`` in its run dir gets that link recreated on the *supervisor's*
disk -- the laptop, or the always-on scheduler droplet -- and every walker that follows links then
reads the supervisor's own private key and uploads it to R2 (mid-run partials, the end-of-run
artifact upload) and from there into ``lab export`` bundles. No walker over a job's output may
follow a link: not a file link, not a directory link, not a link to a file *inside* the dir.

(The rsync side -- ``--no-links`` in ``sky_runner._rsync_down`` -- is a separate defence.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.partials import PartialMirror, eligible
from lab.storage import R2Store

SECRET = b"-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAB3NzaC1yc2E\n-----END OPENSSH PRIVATE KEY-----\n"


class FakeR2:
    def __init__(self) -> None:
        self.objs: dict[str, bytes] = {}

    def put_bytes(self, key: str, data: bytes) -> None:
        self.objs[key] = data

    def put_text(self, key: str, text: str) -> None:
        self.objs[key] = text.encode()

    def uri(self, p: str) -> str:
        return f"r2://b/{p}"


class FakeS3:
    def __init__(self) -> None:
        self.objs: dict[str, bytes] = {}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        self.objs[Key] = Path(Filename).read_bytes()


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """An output dir as rsync leaves it after a hostile box planted links in it."""
    secret = tmp_path / "home" / ".ssh" / "sky-key"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(SECRET)
    out = tmp_path / "runs" / "j-1" / "output"
    out.mkdir(parents=True)
    (out / "results.csv").symlink_to(secret)  # a file link out of the dir
    (out / "real.csv").write_bytes(b"seed,acc\n0,0.9\n")
    (out / "alias.csv").symlink_to(out / "real.csv")  # a link that stays inside the dir
    (out / "ssh").symlink_to(secret.parent, target_is_directory=True)  # a directory link
    nested = out / "sub"
    nested.mkdir()
    (nested / "metrics.jsonl").symlink_to(secret)
    (nested / "ok.txt").write_bytes(b"fine\n")
    return out


def test_partials_never_mirror_a_symlink(planted: Path) -> None:
    r2 = FakeR2()
    PartialMirror("j-1", planted, r2_factory=lambda: r2, interval_s=0).maybe_mirror("running")

    assert all(SECRET not in body for body in r2.objs.values())
    assert "j-1/_partial/results.csv" not in r2.objs
    assert "j-1/_partial/alias.csv" not in r2.objs  # even one pointing inside the dir
    assert r2.objs["j-1/_partial/real.csv"] == b"seed,acc\n0,0.9\n"  # real files still go


def test_eligible_lists_no_symlink(planted: Path) -> None:
    assert [p.name for p in eligible(planted)] == ["real.csv"]


def test_upload_dir_never_follows_a_symlink(planted: Path) -> None:
    s3 = FakeS3()
    count = R2Store("https://x", "b", client=s3).upload_dir(planted, "j-1")

    assert all(SECRET not in body for body in s3.objs.values())
    assert sorted(s3.objs) == ["j-1/real.csv", "j-1/sub/ok.txt"]
    assert count == 2


def test_upload_dir_does_not_descend_into_a_linked_directory(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "id_ed25519").write_bytes(SECRET)
    out = tmp_path / "out"
    out.mkdir()
    (out / "linked").symlink_to(elsewhere, target_is_directory=True)
    s3 = FakeS3()

    assert R2Store("https://x", "b", client=s3).upload_dir(out, "j-1") == 0
    assert s3.objs == {}


def test_a_link_swapped_in_after_listing_is_not_read(tmp_path: Path, monkeypatch) -> None:
    """The listing and the read are two steps; rsync runs beside the mirror, so the file can be
    replaced by a link in between. The read itself must refuse to follow one."""
    secret = tmp_path / "sky-key"
    secret.write_bytes(SECRET)
    out = tmp_path / "out"
    out.mkdir()
    victim = out / "results.csv"
    victim.write_bytes(b"seed\n0\n")
    import lab.partials as P

    real_eligible = P.eligible

    def eligible_then_swap(d: Path) -> list[Path]:
        found = real_eligible(d)
        victim.unlink()
        victim.symlink_to(secret)
        return found

    monkeypatch.setattr(P, "eligible", eligible_then_swap)
    r2 = FakeR2()
    PartialMirror("j-1", out, r2_factory=lambda: r2, interval_s=0).maybe_mirror("running")

    assert all(SECRET not in body for body in r2.objs.values())
