"""DigitalOcean's provision-failure text was boilerplate that pushed the real cause off the end.

Field report 2026-08-23. Five DO jobs failed within 14 seconds of each other and all five landed
on the manifest as the *same* 300 characters, none of which say what went wrong::

    if this is a DigitalOcean setup issue, check `sky check` shows DO enabled (doctl token at
    ~/.config/doctl/config.yaml) and your DO vCPU quota covers the size — launch error: Failed
    to provision all possible launchable resources. Relax the task's resource requirements: 1x
    DO(cpus=4, disk_siz

`end_reason` is capped at 300 chars, and `provision_failure_reason`'s docstring is explicit that
this is why the diagnosis must lead: SkyPilot's generic message alone exceeds the cap. GCP honours
that — `_gcp_failure_hint` reads the error and returns a cause-specific sentence. DO's branch
returns one fixed 158-character string no matter what happened, so it spends half the budget
saying nothing and truncates the provider's own words to make room.

The cost is not cosmetic. Reviewing this incident, the DO failures were the one thing the ledger
could not explain -- whether it was the account droplet limit, a size restriction, or genuine
capacity was unrecoverable after the fact, because the only copy of the answer had been cut off.

**Markers here are taken from the run's own log, not from memory.** The observed text
(`runs/20260823-104026-9460f6/logs.txt`) is::

    sky.exceptions.ResourcesUnavailableError: Failed to acquire resources in all zones in nyc1
    for {DO(cpus=4, disk_size=50)}.

repeated for nyc1/nyc2/nyc3/sfo1/sfo2/sfo3, ~2-3 seconds apart. Deliberately *not* added: markers
for the 422 "size restricted" / "invalid size specified" tier responses. Those are real (they are
why the CPU defaults are 4 vCPU + 50 GB) but DO's own wording for them is not in any log on this
machine, and inventing a marker string that never matches is worse than no marker at all.
"""

from __future__ import annotations

import pytest

from lab.sky_runner import provision_failure_reason

# The cap `end_reason` is stored under (`store.update_manifest(..., end_reason=reason[:300])`).
END_REASON_CAP = 300

# Verbatim from the failed run's log, wrapped the way sky hands it to the supervisor.
LIVE_DO_ERROR = (
    "launch error: Failed to provision all possible launchable resources. Relax the task's "
    "resource requirements: 1x DO(cpus=4, disk_size=50). Failed to acquire resources in all "
    "zones in nyc1 for {DO(cpus=4, disk_size=50)}."
)


class TestTheSpecificCauseSurvivesTruncation:
    def test_the_provider_text_is_not_truncated_away(self) -> None:
        """The regression: what reaches the manifest must still identify the failure."""
        stored = provision_failure_reason(LIVE_DO_ERROR, "do")[:END_REASON_CAP]

        assert "acquire resources in all zones" in stored, (
            f"the distinguishing part of DO's error did not survive the {END_REASON_CAP}-char "
            f"cap; stored was:\n{stored}"
        )

    def test_the_diagnosis_leads(self) -> None:
        """`provision_failure_reason`'s stated contract: the actionable sentence goes first."""
        reason = provision_failure_reason(LIVE_DO_ERROR, "do")
        assert not reason.startswith("launch error:")

    def test_an_all_zones_failure_is_diagnosed_not_boilerplated(self) -> None:
        """A cause we *can* name must be named, as GCP already does."""
        reason = provision_failure_reason(LIVE_DO_ERROR, "do")
        low = reason.lower()
        assert "droplet" in low or "limit" in low, (
            f"all-zones exhaustion on DO has a known shortlist of causes; say them: {reason}"
        )

    def test_an_unrecognised_do_error_still_leaves_room_for_the_text(self) -> None:
        """The fallback hint must not eat the budget either -- that was the whole defect."""
        odd = "launch error: " + "some unrecognised DigitalOcean failure mode. " * 6
        stored = provision_failure_reason(odd, "do")[:END_REASON_CAP]
        assert "unrecognised DigitalOcean failure mode" in stored, stored

    @pytest.mark.parametrize("cloud", ["do", "gcp"])
    def test_no_hint_may_crowd_out_the_provider(self, cloud: str) -> None:
        """The invariant, for every diagnosing cloud: advice never costs us the evidence.

        Stated as a floor rather than a hint-length limit on purpose. Some hints are long
        because they are genuinely worth their space -- GCP's two-level GPU-quota explanation
        cost a live launch to learn -- so the rule is that the provider's own words are
        guaranteed a share of the budget, not that advice must be terse.
        """
        stored = provision_failure_reason(LIVE_DO_ERROR, cloud)[:END_REASON_CAP]

        # Asserted on the provider's actual closing words rather than on whatever follows the
        # separator: a hint containing its own " — " would otherwise satisfy a split-based check
        # while the evidence it was supposed to protect had been truncated away.
        assert LIVE_DO_ERROR[-60:] in stored, (
            f"{cloud} truncated the end of the provider's message away: {stored}"
        )


class TestOtherCloudsAreUnchanged:
    def test_an_unknown_cloud_passes_the_text_through(self) -> None:
        assert provision_failure_reason("launch error: whatever", "somecloud") == (
            "launch error: whatever"
        )
