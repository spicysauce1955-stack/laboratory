"""Two small CLI defects that cost real time during the 2026-08-19/20 incident (F6, F7).

Neither costs money. Both cost *trust in the record*, which during an incident is nearly as
expensive -- the ledger is the thing you reason from when the machines are gone.

**F6** was reported as "`lab <cmd> --help` exits non-deterministically 0 or 1". It is not
non-deterministic at all; it is an unhandled ``BrokenPipeError``:

    lab submit --help              -> 0   (20 of 20 runs)
    lab submit --help | head -3    -> 1   ( 5 of  5 runs)

The ledger's mixed statuses for the 31 recorded `submit --help` calls record nothing but whether
the caller piped the output into something that closed early. Every one of those `error` rows is a
false failure sitting in `lab history` and `lab report`, indistinguishable from a real one without
inspecting the argv. Standard CLI behaviour is to treat a closed stdout as a clean exit.

**F7**: after a burst of 13 DO failures on 2026-08-19, something tried ``lab kill <job_id>`` 19
times across those 13 job ids. `kill` is not a command -- `cancel` is -- so every attempt exited 2
with `No such command 'kill'.` and no suggestion. **None of those 13 jobs was ever cancelled
through the tool.**

The field report assumed click offers no suggestions. It does: ``lab stat`` already answers
"No such command 'stat'. Did you mean 'status'?", and the near-miss cases below pass unchanged.
The real gap is narrower and not fixable by string distance -- `kill` is not a *typo* for
`cancel`, it is what the rest of the world calls that operation. Only an explicit synonym table
can bridge that, and only the synonyms someone would actually reach for belong in it.

The suggestion deliberately does not *run* the aliased command. Every alias here maps onto a
destructive operation, and silently reinterpreting one is a worse failure than the one being
fixed.
"""

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from lab.cli import app

runner = CliRunner()


def _lab(*args, **kwargs):
    """Run the real console entry point in a subprocess.

    Deliberately not CliRunner: both defects live in process-level plumbing -- a real closed pipe
    and the real `lab.cli:main` exit path -- which an in-process runner with a fake stdout cannot
    reproduce. This is the only way these tests can fail for the right reason.
    """
    return subprocess.run(
        [sys.executable, "-c", "from lab.cli import main; main()", *args],
        capture_output=True,
        **kwargs,
    )


class TestClosedStdoutPipeIsCleanExit:
    def test_help_into_a_closed_pipe_exits_0(self):
        """The exact reproduction from the ledger: `lab submit --help | head -3`."""
        helper = subprocess.Popen(
            [sys.executable, "-c", "from lab.cli import main; main()", "submit", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert helper.stdout is not None
        helper.stdout.readline()
        helper.stdout.close()  # the `| head -3` moment

        assert helper.wait(timeout=60) == 0

    def test_help_without_a_pipe_still_exits_0(self):
        assert _lab("submit", "--help").returncode == 0

    def test_a_broken_pipe_does_not_print_a_traceback(self):
        """A traceback on stderr is noise an operator has to read past mid-incident."""
        helper = subprocess.Popen(
            [sys.executable, "-c", "from lab.cli import main; main()", "submit", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert helper.stdout is not None
        helper.stdout.readline()
        helper.stdout.close()
        _, stderr = helper.communicate(timeout=60)

        assert b"BrokenPipeError" not in stderr
        assert b"Traceback" not in stderr

    def test_a_real_usage_error_still_exits_2(self):
        """No regression: silencing a broken pipe must not silence genuine failures."""
        assert _lab("definitely-not-a-command").returncode == 2


class TestUnknownCommandSuggests:
    def test_lab_kill_names_cancel(self):
        """The one that cost 13 uncancelled jobs."""
        result = _lab("kill", "20260820-071905-771110")

        assert result.returncode == 2
        combined = (result.stdout + result.stderr).decode()
        assert "cancel" in combined, f"no suggestion offered: {combined!r}"

    @pytest.mark.parametrize(
        "typo,expected",
        [
            ("stat", "status"),
            ("reconile", "reconcile"),
            ("sweap", "sweep"),
            ("hisory", "history"),
        ],
    )
    def test_near_misses_are_suggested(self, typo, expected):
        combined = (lambda r: (r.stdout + r.stderr).decode())(_lab(typo))

        assert expected in combined

    def test_a_wild_miss_suggests_nothing_rather_than_guessing(self):
        """A confident wrong suggestion is worse than none -- it sends you down a blind alley."""
        result = _lab("zzzzzzzzzz")

        assert result.returncode == 2
        combined = (result.stdout + result.stderr).decode()
        assert "did you mean" not in combined.lower()

    @pytest.mark.parametrize("alias,real", [("kill", "cancel"), ("stop", "cancel")])
    def test_semantic_aliases_are_bridged(self, alias, real):
        """Not typos -- other tools' names for the same operation. String distance cannot help."""
        combined = (lambda r: (r.stdout + r.stderr).decode())(_lab(alias, "some-job-id"))

        assert real in combined

    def test_an_alias_is_suggested_not_executed(self, tmp_path):
        """Silently reinterpreting a destructive command is worse than the bug being fixed."""
        result = _lab("kill", "some-job-id")

        assert result.returncode == 2
        assert b"cancelled" not in result.stdout.lower()

    def test_a_valid_command_is_unaffected(self):
        result = runner.invoke(app, ["list"])

        assert "No such command" not in result.output
