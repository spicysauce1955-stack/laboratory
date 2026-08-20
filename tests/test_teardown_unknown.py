"""Teardown needs a third state: "I could not read the answer" (R10).

`teardown_status` had two values, chosen by one line -- ``"succeeded" if succeeded else "failed"``.
So when the outcome is genuinely unreadable, the code had to pick one, and it picked the alarm.
On 2026-08-20 seven teardowns recorded ``failed`` while all seven machines had in fact been
destroyed: a 100% false alarm rate on the one signal FR-C2 exists to raise.

That is corrosive in a specific way. `failed` is meant to mean *drop everything, a machine is
still billing*. If most `failed` values are really "the client could not decode the reply", an
operator learns to discount exit 3 -- and the next real leak goes unnoticed. The ledger already
shows the beginning of that: `lab reconcile` was run 12 times in 15 hours because the signal could
not be trusted.

The contract these tests pin:

    succeeded  the machine is confirmed gone                      lab wait -> 0
    failed     the destroy was definitively refused: a real leak   lab wait -> 3   (unchanged)
    unknown    the outcome could not be read; verify with cloud    lab wait -> 6   (new)
    None       no teardown was ever recorded                       lab wait -> 0 + warning

`failed` outranks `unknown`, which outranks `--fail-fast`'s exit 4: a *confirmed* money alarm is
the most urgent thing `lab wait` can report, and an unverifiable one is still a money signal.
"""

import json

import pytest
from helpers import make_manifest
from typer.testing import CliRunner

import lab.cli as cli_mod
from lab.backends import skypilot as m
from lab.cli import app
from lab.models import BackendInfo, JobState

runner = CliRunner()

# The exact live signature: the server destroyed the cluster, the client could not decode the reply.
UNDECODABLE = AttributeError(
    "Can't get attribute 'user_initiated_down' on <module 'sky.core' from '/x/sky/core.py'>"
)


class _ClusterGone(Exception):
    pass


_ClusterGone.__name__ = "ClusterDoesNotExist"


class _Sky:
    def __init__(self, exc):
        self.exc = exc

    def get(self, x):
        return x

    def down(self, cluster):
        raise self.exc


class TestTeardownRecordsUnknown:
    """`unknown` is for when *nothing* can settle the question -- not merely when sky stumbled.

    Where a provider-direct fallback exists it is the tiebreaker: it asks DigitalOcean/GCP/Vast
    itself, and their answer is authoritative regardless of what the sky client could or could not
    decode. So an undecodable reply plus a fallback that looks and finds nothing is `succeeded`
    (positively verified gone), not `unknown`. Reaching for `unknown` there would be the same
    mistake as `failed` in the other direction: manufacturing doubt we do not actually have.
    """

    def test_an_undecodable_reply_with_no_way_to_verify_is_unknown(self, tmp_path, monkeypatch):
        """The 2026-08-20 case on a cloud with no direct fallback. `failed` here is a lie."""
        store = _store_with_job(tmp_path, "t1", cloud="lambda")

        m.tear_down_and_record(_Sky(UNDECODABLE), "lab-t1", store, "t1", "lambda", backoffs=())

        assert store.read_manifest("t1").teardown_status == "unknown"

    def test_an_undecodable_reply_the_provider_can_settle_is_succeeded(self, tmp_path, monkeypatch):
        """DO answered: nothing matching is running. That is knowledge, not doubt."""
        store = _store_with_job(tmp_path, "t1b")
        monkeypatch.setattr(m, "_do_destroy_matching", lambda c: ([], []))

        m.tear_down_and_record(_Sky(UNDECODABLE), "lab-t1b", store, "t1b", "do", backoffs=())

        assert store.read_manifest("t1b").teardown_status == "succeeded"

    def test_an_undecodable_reply_whose_fallback_also_fails_is_unknown(self, tmp_path, monkeypatch):
        """sky could not tell us and the provider could not either -- genuinely unknown."""
        store = _store_with_job(tmp_path, "t1c")

        def _unreachable(cluster):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(m, "_do_destroy_matching", _unreachable)

        m.tear_down_and_record(_Sky(UNDECODABLE), "lab-t1c", store, "t1c", "do", backoffs=())

        assert store.read_manifest("t1c").teardown_status == "unknown"

    def test_a_definitive_refusal_whose_fallback_fails_is_still_failed(self, tmp_path, monkeypatch):
        """We know sky did not do it, and we know we could not either. That is a real alarm."""
        store = _store_with_job(tmp_path, "t1d")

        def _unreachable(cluster):
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(m, "_do_destroy_matching", _unreachable)

        m.tear_down_and_record(
            _Sky(_ClusterGone("gone")), "lab-t1d", store, "t1d", "do", backoffs=()
        )

        assert store.read_manifest("t1d").teardown_status == "failed"

    def test_a_definitive_refusal_still_records_failed(self, tmp_path, monkeypatch):
        """`failed` must stay meaningful -- it is the thing exit 3 is for."""
        store = _store_with_job(tmp_path, "t2")
        monkeypatch.setattr(m, "_do_destroy_matching", lambda c: ([], ["droplet lab-t2: 422"]))

        m.tear_down_and_record(_Sky(_ClusterGone("gone")), "lab-t2", store, "t2", "do", backoffs=())

        assert store.read_manifest("t2").teardown_status == "failed"

    def test_success_is_unchanged(self, tmp_path, monkeypatch):
        """A clean `sky.down` whose storage is *verified* gone still records `succeeded`.

        The verification is not optional on DO: since P4-a a clean `sky.down` alone is not proof,
        so this test has to say what the volume sweep found. Stubbing it to "nothing left" is the
        success case; leaving it unstubbed would (correctly) yield `unknown`.
        """
        store = _store_with_job(tmp_path, "t3")
        monkeypatch.setattr(m, "_do_sweep_leftover_volumes", lambda cluster: ([], []))

        class _OkSky:
            def get(self, x):
                return x

            def down(self, cluster):
                return None

        m.tear_down_and_record(_OkSky(), "lab-t3", store, "t3", "do", backoffs=())

        assert store.read_manifest("t3").teardown_status == "succeeded"

    def test_the_unknown_end_reason_says_how_to_settle_it(self, tmp_path, monkeypatch):
        """An `unknown` the reader cannot act on is no better than a wrong answer."""
        store = _store_with_job(tmp_path, "t4", cloud="lambda")

        m.tear_down_and_record(_Sky(UNDECODABLE), "lab-t4", store, "t4", "lambda", backoffs=())

        reason = store.read_manifest("t4").end_reason or ""
        assert "unknown" in reason.lower()
        assert "verify" in reason.lower()


def _store_with_job(tmp_path, job_id, *, cloud="do"):
    from lab.store import JobStore

    store = JobStore(tmp_path)
    mf = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
        update={"status": JobState.running, "backend": BackendInfo(provisioner="skypilot")}
    )
    mf.resources.cloud = cloud
    store.create(mf)
    return store


# ---------------------------------------------------------------------------
# `lab wait` exit codes
# ---------------------------------------------------------------------------

_BASE = {
    "all_terminal": True,
    "failed_fast": False,
    "pending": [],
    "teardown_leaks": [],
    "teardown_unknown": [],
    "teardown_unconfirmed": [],
    "jobs": [],
}


def _patch_wait(monkeypatch, tmp_path, **over):
    """Wire `lab wait` to a canned summary, mirroring tests/test_cli_wait.py's harness.

    The manifest path must merely *exist* -- `wait` guards against unknown job ids before it
    reaches any of the exit-code logic under test here.
    """
    summary = {**_BASE, **over}

    class _FakeLab:
        def wait_summary(self, ids, **kw):
            return summary

    class _FakeStore:
        def __init__(self, home):
            pass

        def manifest_path(self, job_id):
            path = tmp_path / f"{job_id}.json"
            path.touch()
            return path

    monkeypatch.setattr(cli_mod, "_lab_for", lambda job_id: _FakeLab())
    monkeypatch.setattr(cli_mod, "JobStore", _FakeStore)
    return summary


class TestWaitExitCodes:
    def test_an_unknown_teardown_exits_6(self, monkeypatch, tmp_path):
        _patch_wait(monkeypatch, tmp_path, teardown_unknown=["j1"])

        result = runner.invoke(app, ["wait", "j1"])

        assert result.exit_code == 6, result.output

    def test_a_confirmed_leak_outranks_an_unknown(self, monkeypatch, tmp_path):
        """Exit 3 is the urgent one; it must never be masked by a weaker signal."""
        _patch_wait(monkeypatch, tmp_path, teardown_leaks=["j1"], teardown_unknown=["j2"])

        result = runner.invoke(app, ["wait", "j1"])

        assert result.exit_code == 3, result.output

    def test_an_unknown_outranks_fail_fast(self, monkeypatch, tmp_path):
        """A money signal outranks the fail-fast signal, same as exit 3 already does."""
        _patch_wait(monkeypatch, tmp_path, failed_fast=True, teardown_unknown=["j1"])

        result = runner.invoke(app, ["wait", "j1", "--fail-fast"])

        assert result.exit_code == 6, result.output

    def test_a_clean_run_still_exits_0(self, monkeypatch, tmp_path):
        _patch_wait(monkeypatch, tmp_path)

        assert runner.invoke(app, ["wait", "j1"]).exit_code == 0

    def test_a_null_teardown_still_only_warns(self, monkeypatch, tmp_path):
        """`None` means nothing was ever recorded -- weaker than `unknown`, unchanged behaviour."""
        _patch_wait(monkeypatch, tmp_path, teardown_unconfirmed=["j1"])

        result = runner.invoke(app, ["wait", "j1"])

        assert result.exit_code == 0, result.output
        assert "not confirmed" in result.stderr

    def test_the_unknown_message_names_the_jobs_and_the_remedy(self, monkeypatch, tmp_path):
        _patch_wait(monkeypatch, tmp_path, teardown_unknown=["j1"])

        result = runner.invoke(app, ["wait", "j1"])

        assert "j1" in result.stderr
        assert "unknown" in result.stderr.lower()
        assert "reconcile" in result.stderr or "verify" in result.stderr.lower()

    def test_stdout_stays_parseable_json(self, monkeypatch, tmp_path):
        _patch_wait(monkeypatch, tmp_path, teardown_unknown=["j1"])

        result = runner.invoke(app, ["wait", "j1"])

        assert json.loads(result.stdout)["teardown_unknown"] == ["j1"]


class TestWaitSummaryClassifies:
    def test_unknown_is_reported_separately_from_leaks_and_nulls(self, tmp_path):
        """All three states must be distinguishable by a caller reading the summary."""
        from lab.backends.local import LocalBackend
        from lab.core import Lab

        lab = Lab(backend=LocalBackend(home=tmp_path, repo=tmp_path), repo=tmp_path, home=tmp_path)
        for job_id, teardown in (("a", "failed"), ("b", "unknown"), ("c", None)):
            store = lab.store
            mf = make_manifest(job_id, "python x.py", timeout="1h").model_copy(
                update={
                    "status": JobState.succeeded,
                    "backend": BackendInfo(provisioner="skypilot"),
                    "teardown_status": teardown,
                }
            )
            store.create(mf)

        summary = lab.wait_summary(["a", "b", "c"], interval=0, timeout=0)

        assert summary["teardown_leaks"] == ["a"]
        assert summary["teardown_unknown"] == ["b"]
        assert summary["teardown_unconfirmed"] == ["c"]


@pytest.mark.parametrize("value", ["succeeded", "failed", "unknown", None])
def test_the_manifest_accepts_every_state(tmp_path, value):
    store = _store_with_job(tmp_path, "tm")
    store.update_manifest("tm", teardown_status=value)
    assert store.read_manifest("tm").teardown_status == value
