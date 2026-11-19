"""A teardown's verdict must carry its evidence, or the next incident is unexplainable.

Job E (2026-11-19) recorded ``teardown_status: "succeeded"`` over a volume that was still billing,
and *nothing* on record said why: ``robust_teardown`` returned the swept volume ids and what it
listed, ``tear_down_and_record`` kept only the status string, and the ledger's notes are flushed
only when a call fails — which a false success, by definition, does not. The mechanism had to be
reconstructed from SkyPilot's source rather than read off the job.

``teardown_detail`` is that evidence, persisted on every teardown, success included: how it was
done, what was destroyed, and for DigitalOcean which volumes were known, discovered, verified gone
by id, deleted by us, refused or left unverified. Additive — a manifest written before it existed
must still load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_do import FakeDo, FakeSky, http_error
from helpers import make_manifest

from lab.backends import skypilot as m
from lab.models import JobManifest, JobState, ResourceRequest
from lab.store import JobStore

CLUSTER = "lab-smoke-20261119-111134-88cf91"
VOLUME = f"{CLUSTER}-3dd12990-ea4c-head"


def _store(tmp_path: Path, job_id: str, cloud: str = "do") -> JobStore:
    store = JobStore(tmp_path / "runs")
    store.create(
        make_manifest(job_id, "python x.py", resources=ResourceRequest(cloud=cloud)).model_copy(
            update={"status": JobState.timed_out}
        )
    )
    return store


def _do(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> FakeDo:
    do = FakeDo(
        droplets=[{"id": 7, "name": VOLUME}],
        volumes=[{"id": "vol-e", "name": VOLUME, "droplet_ids": [7]}],
        **kw,
    )
    monkeypatch.setattr(m, "_get_do_client", lambda: do)
    monkeypatch.setattr(m.time, "sleep", do.sleep)
    return do


class TestRecordedOnSuccess:
    def test_a_clean_do_teardown_records_how_the_volume_was_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _do(monkeypatch, detach_s=0.0, associated_delete=False)
        store = _store(tmp_path, "e1")
        store.write_runtime("e1", do_volume_ids=["vol-e"])

        assert m.tear_down_and_record(FakeSky(do), CLUSTER, store, "e1", "do", backoffs=(0,))

        detail = store.read_manifest("e1").teardown_detail
        assert detail is not None
        assert detail["status"] == "succeeded" and detail["cloud"] == "do"
        assert detail["via"] == "sky.down" and detail["attempts"] == 1
        assert detail["at"]
        vols = detail["do_volumes"]
        assert vols["verification"] == "get-by-id"
        assert vols["known"] == ["vol-e"] and vols["gone"] == ["vol-e"]
        assert vols["deleted"] == ["vol-e"], "we, not DO's background job, removed it"

    def test_a_vast_teardown_records_its_shape_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store(tmp_path, "v1", cloud="vast")
        monkeypatch.setattr(m, "_vast_destroy_matching", lambda c: ([123], []))

        class _DownFails:
            def get(self, x: Any) -> Any:
                return x

            def down(self, cluster: str) -> None:
                raise RuntimeError("API server 503")

        m.tear_down_and_record(_DownFails(), "lab-v1", store, "v1", "vast", backoffs=())

        detail = store.read_manifest("v1").teardown_detail
        assert detail is not None
        assert detail["via"] == "vast-direct" and detail["destroyed"] == [123]
        assert "503" in detail["error"]


class TestRecordedOnFailure:
    def test_a_refused_volume_is_named_in_the_detail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        do = _do(monkeypatch, detach_s=0.0, associated_delete=False)
        do.refuse["vol-e"] = http_error(403, "You do not have access", error_id="forbidden")
        store = _store(tmp_path, "e2")

        assert not m.tear_down_and_record(FakeSky(do), CLUSTER, store, "e2", "do", backoffs=(0,))

        manifest = store.read_manifest("e2")
        assert manifest.teardown_status == "failed"
        detail = manifest.teardown_detail
        assert detail is not None and detail["status"] == "failed"
        assert "vol-e" in detail["do_volumes"]["refused"]


class TestCompatibility:
    def test_a_manifest_written_before_the_field_existed_still_loads(self, tmp_path: Path) -> None:
        legacy = make_manifest("old", "python x.py").model_dump(mode="json")
        legacy.pop("teardown_detail", None)
        legacy["teardown_status"] = "succeeded"

        loaded = JobManifest.model_validate(legacy)

        assert loaded.teardown_detail is None and loaded.teardown_status == "succeeded"

    def test_the_detail_stays_compact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A manifest is read on every `lab status`; the evidence must not swamp it."""
        store = _store(tmp_path, "big", cloud="vast")
        monkeypatch.setattr(m, "_vast_destroy_matching", lambda c: (list(range(500)), []))

        class _DownFails:
            def get(self, x: Any) -> Any:
                return x

            def down(self, cluster: str) -> None:
                raise RuntimeError("x" * 5000)

        m.tear_down_and_record(_DownFails(), "lab-big", store, "big", "vast", backoffs=())

        detail = store.read_manifest("big").teardown_detail
        assert detail is not None
        assert len(detail["destroyed"]) <= 50 and len(detail["error"]) <= 600
