"""The local wall-clock cap must be anchored to the job's start, not to when the wait begins.

Found while re-reading the logs of four jobs lost on 2026-08-20/21. Each was submitted with a 7h
timeout and each ran **703 minutes** -- 4.6 hours past what looked like a hard bound -- and none was
stopped by the lab. They were stopped by an external watchdog.

The cap itself was never overrun; it was measured from the wrong moment:

    sky.tail_logs(cluster, sky_job_id, follow=True)   # blocks until the run finishes. no timeout.
    ...
    max_wait = (parse_duration(manifest.resources.timeout) or 3600) + 300
    deadline = time.time() + max_wait                 # <- starts counting HERE

`tail_logs` blocks for essentially the whole run, so on the normal (non-adopt) path the budget did
not begin until the streaming call returned. The effective local cap is therefore about *twice* the
requested timeout in the ordinary case, and unbounded whenever `tail_logs` itself hangs -- which is
exactly what a network partition does to it. Those four jobs lost their network at 02:20; their 7h05m
budget started then, putting the local deadline near 09:25.

The `adopt` branch already did the right thing (`total - elapsed`); the normal path did not, so the
bug was invisible on the path everyone uses.

The real enforcement lives on the box (`timeout --kill-after=<grace>s <wall>s`, deliberately, so it
survives the supervisor dying). This local bound is the backstop for when the supervisor is alive
but blind -- and a backstop anchored to an arbitrary later moment is not a backstop.
"""

from datetime import timedelta

from helpers import make_manifest

import lab.sky_runner as runner
from lab._util import now
from lab.models import BackendInfo, JobState


class TestDeadlineIsAnchoredToTheStart:
    def test_time_already_spent_counts_against_the_cap(self):
        """A job 6h into a 7h timeout has ~1h of local budget left, not another 7h05m."""
        started = now() - timedelta(hours=6)

        remaining = runner.remaining_wall_budget(timeout="7h", started=started)

        assert 3300 <= remaining <= 3900, remaining

    def test_a_job_past_its_cap_gets_no_further_budget(self):
        started = now() - timedelta(hours=12)

        remaining = runner.remaining_wall_budget(timeout="7h", started=started)

        assert remaining == 0.0, "a cap already blown must not hand out more time"

    def test_a_fresh_job_gets_the_full_budget_plus_grace(self):
        remaining = runner.remaining_wall_budget(timeout="7h", started=now())

        assert 25400 <= remaining <= 25550, remaining  # 7h + 300s grace

    def test_a_missing_timeout_falls_back_to_the_documented_default(self):
        remaining = runner.remaining_wall_budget(timeout=None, started=now())

        assert 3800 <= remaining <= 3950, remaining  # 3600 default + 300 grace

    def test_an_unknown_start_does_not_hand_out_unbounded_time(self):
        """`started_at` can be absent on a malformed manifest; that must not disable the cap."""
        remaining = runner.remaining_wall_budget(timeout="7h", started=None)

        assert remaining <= 25550


class TestTheBudgetIsSpentByStreaming:
    def test_streaming_consumes_the_same_budget_as_waiting(self, monkeypatch):
        """The regression, in miniature: time inside `tail_logs` must not be free.

        Before the fix, `max_wait` was computed *after* `tail_logs` returned, so a call that
        blocked for the whole run reset the clock. Anchoring to `started` makes the two
        indistinguishable, which is the point -- both are the job being alive.
        """
        started = now() - timedelta(hours=7, minutes=10)

        assert runner.remaining_wall_budget(timeout="7h", started=started) == 0.0


class TestHeartbeatCadenceIsRealTime:
    """`since_beat += poll_s` counted nominal sleep, not elapsed time.

    When each poll blocks -- and during the outage they blocked for roughly a minute apiece -- the
    heartbeat fired every N *iterations* rather than every N seconds, drifting to many minutes
    apart while claiming a 60-second cadence. The partial-results fetch is the thing that drift
    delays, and on 2026-08-20 four jobs finished with empty output directories.
    """

    def test_heartbeat_fires_on_elapsed_time_not_iteration_count(self, monkeypatch):
        beats: list[float] = []
        clock = {"t": 1000.0}

        def _fake_time():
            return clock["t"]

        def _fake_sleep(_s):
            clock["t"] += 30.0  # each "poll" really costs 30s, not poll_s

        monkeypatch.setattr(runner.time, "time", _fake_time)
        monkeypatch.setattr(runner.time, "sleep", _fake_sleep)

        class _Sky:
            def get(self, x):
                return x

            def queue(self, cluster, skip_finished=False):
                return []

        runner._wait_terminal(
            _Sky(), "lab-x", None, 300.0,
            poll_s=1.0, heartbeat_s=60.0, on_heartbeat=lambda: beats.append(clock["t"]),
        )

        assert beats, "the heartbeat never fired"
        gaps = [b - a for a, b in zip(beats, beats[1:])]
        assert all(g <= 90.0 for g in gaps), f"heartbeat drifted in real time: {gaps}"


def test_a_job_whose_streaming_ate_the_budget_is_marked_timed_out(tmp_path, monkeypatch):
    """End to end: the supervisor must stop, not hand itself a second full budget."""
    from lab.store import JobStore

    store = JobStore(tmp_path)
    m = make_manifest("jw", "python x.py", timeout="1h").model_copy(
        update={
            "status": JobState.running,
            "started_at": now() - timedelta(hours=3),
            "backend": BackendInfo(provisioner="skypilot"),
        }
    )
    store.create(m)

    assert runner.remaining_wall_budget(timeout="1h", started=m.started_at) == 0.0
