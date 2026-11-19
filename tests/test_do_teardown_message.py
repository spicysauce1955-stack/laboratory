"""A DO teardown alarm must describe what actually happened and what to run next.

The ``failed`` annotation on DigitalOcean still read "(no provider-direct fallback for do)" — true
before F2, false since: a DO-direct fallback (droplet + volume through the DO API) has existed for
months. On job F (2026-11-19) the failure was not even about the fallback: ``sky.down`` had
destroyed the droplet, and only the volume check failed. An operator reading "no fallback" is told
the lab has fewer tools than it does, and is pointed at the droplet rather than at the volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_do import FakeDo, FakeSky, http_error
from helpers import make_manifest

from lab.backends import skypilot as m
from lab.models import JobState, ResourceRequest
from lab.store import JobStore

CLUSTER = "lab-smoke-20261119-111134-b048d2"
VOLUME = f"{CLUSTER}-3dd12990-ef65-head"


def _store(tmp_path: Path, job_id: str, cloud: str = "do") -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(
        make_manifest(job_id, "python x.py", resources=ResourceRequest(cloud=cloud)).model_copy(
            update={"status": JobState.cancelled}
        )
    )
    return store


def _do(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> FakeDo:
    do = FakeDo(
        droplets=[{"id": 31, "name": VOLUME}],
        volumes=[{"id": "vol-f", "name": VOLUME, "droplet_ids": [31]}],
        detach_s=0.0,
        associated_delete=False,
        **kw,
    )
    monkeypatch.setattr(m, "_get_do_client", lambda: do)
    monkeypatch.setattr(m.time, "sleep", do.sleep)
    return do


class TestDoAlarmWording:
    def test_a_volume_left_after_a_clean_sky_down_names_the_volume_not_a_missing_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _do(monkeypatch)
        do.refuse["vol-f"] = http_error(403, "You do not have access", error_id="forbidden")
        store = _store(tmp_path, "f1")

        m.tear_down_and_record(FakeSky(do), CLUSTER, store, "f1", "do", backoffs=(0,))

        reason = store.read_manifest("f1").end_reason or ""
        assert "no provider-direct fallback" not in reason
        assert "vol-f" in reason and "volume" in reason
        assert "lab reconcile --apply" in reason

    def test_a_failed_do_direct_fallback_says_it_ran(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _do(monkeypatch)
        do.droplets.destroy = lambda droplet_id, **kw: (_ for _ in ()).throw(
            http_error(422, "droplet is locked")
        )
        store = _store(tmp_path, "f2")

        m.tear_down_and_record(
            FakeSky(do, down_error=RuntimeError("API server 503")),
            CLUSTER,
            store,
            "f2",
            "do",
            backoffs=(),
        )

        manifest = store.read_manifest("f2")
        assert manifest.teardown_status == "failed"
        reason = manifest.end_reason or ""
        assert "no provider-direct fallback" not in reason
        assert "do-direct fallback" in reason

    def test_a_successful_do_direct_fallback_is_annotated_like_vast_and_gcp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _do(monkeypatch)
        store = _store(tmp_path, "f3")

        m.tear_down_and_record(
            FakeSky(do, down_error=RuntimeError("API server 503")),
            CLUSTER,
            store,
            "f3",
            "do",
            backoffs=(),
        )

        manifest = store.read_manifest("f3")
        assert manifest.teardown_status == "succeeded"
        assert "do-direct fallback destroyed" in (manifest.end_reason or "")

    def test_a_cloud_with_no_fallback_still_says_so(self, tmp_path: Path) -> None:
        store = _store(tmp_path, "l1", cloud="lambda")

        class ClusterDoesNotExist(Exception):
            """A definitive refusal (matched by type name)."""

        class _Sky:
            def get(self, x: Any) -> Any:
                return x

            def down(self, cluster: str) -> None:
                raise ClusterDoesNotExist("gone")

        m.tear_down_and_record(_Sky(), "lab-l1", store, "l1", "lambda", backoffs=())

        assert "no provider-direct fallback for lambda" in (
            store.read_manifest("l1").end_reason or ""
        )
