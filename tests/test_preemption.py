from lab.models import JobState
from lab.preemption import classify_terminal


def base(**kw):
    d = dict(sky_state=JobState.failed, timed_out=False, cancel_requested=False,
             use_spot=False, cluster_gone=False, reached_terminal=False)
    d.update(kw)
    return d


def test_timeout_wins_over_everything():
    assert classify_terminal(**base(timed_out=True, use_spot=True, cluster_gone=True,
                                    cancel_requested=True, reached_terminal=True,
                                    sky_state=JobState.succeeded)) is JobState.timed_out


def test_user_cancel_beats_inferred_preemption():
    assert classify_terminal(**base(cancel_requested=True, use_spot=True,
                                    cluster_gone=True)) is JobState.cancelled


def test_real_success_is_trusted():
    assert classify_terminal(**base(sky_state=JobState.succeeded,
                                    reached_terminal=True)) is JobState.succeeded


def test_real_failure_is_trusted_even_on_spot():
    # job ran and exited non-zero (sky reported FAILED); cluster later gone from teardown
    assert classify_terminal(**base(sky_state=JobState.failed, reached_terminal=True,
                                    use_spot=True, cluster_gone=True)) is JobState.failed


def test_spot_cluster_vanished_without_terminal_is_preempted():
    assert classify_terminal(**base(use_spot=True, cluster_gone=True,
                                    reached_terminal=False)) is JobState.preempted


def test_on_demand_cluster_gone_is_failed_not_preempted():
    assert classify_terminal(**base(use_spot=False, cluster_gone=True)) is JobState.failed
