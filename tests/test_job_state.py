from lab.models import JobState
from lab.core import _TERMINAL_STATES


def test_preempted_is_a_terminal_state():
    assert JobState.preempted.value == "preempted"
    assert JobState.preempted in _TERMINAL_STATES
