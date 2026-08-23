"""`--price-cap` promised a ceiling it does not hold, and nothing ever checked it.

On 2026-08-23 three of nine Vast jobs submitted with `--price-cap 0.85` billed above it, two at
2.61x ($2.220/hr). The two that finished cost $5.50 against an expected ~$1.03.

The cap reaches exactly one place in the launch path -- `sky.Resources(max_hourly_cost=)` -- and
SkyPilot's optimizer honours it against its own catalog. This repo already documents what that
catalog is worth on Vast, in `skypilot.vast_hourly_for_cluster`:

    SkyPilot's ``get_cost()`` reads its own catalog and under-reports Vast prices (~4x low)

A 2.61x overrun sits squarely inside that error band. Meanwhile `resolve_cost` reads the real
`dph_total` seconds after the host is UP -- it is the only reason those overruns are visible in
the manifests at all -- and nothing compared it back to the cap.

This module covers the two pieces that need no cloud to decide: the help text's claim, and the
predicates themselves.
"""

from __future__ import annotations

import pytest
import typer

from lab.cli import app
from lab.pricing import OVER_CAP_TOLERANCE, cap_admission_error, exceeds_cap, over_cap_warning


def _price_cap_helps() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for cmd in app.registered_commands:
        name = cmd.name or (cmd.callback.__name__ if cmd.callback else "?")
        for param in getattr(cmd.callback, "__defaults__", None) or ():
            if not isinstance(param, typer.models.OptionInfo):
                continue
            decls = [d for d in (param.param_decls or ()) if isinstance(d, str)]
            if "--price-cap" in decls:
                found.append((name, param.help or ""))
    return found


class TestTheHelpTextTellsTheTruth:
    def test_some_command_documents_price_cap(self) -> None:
        assert _price_cap_helps(), "no --price-cap help found to check"

    @pytest.mark.parametrize("command,text", _price_cap_helps())
    def test_it_does_not_promise_a_hard_ceiling(self, command: str, text: str) -> None:
        """The claim that cost $4.47 in one afternoon."""
        assert "worst case is a ceiling" not in text.lower(), (
            f"`lab {command} --help` promises a ceiling SkyPilot cannot hold on Vast: {text!r}"
        )

    @pytest.mark.parametrize("command,text", _price_cap_helps())
    def test_it_names_what_the_cap_is_priced_against(self, command: str, text: str) -> None:
        low = text.lower()
        assert "catalog" in low or "estimate" in low, (
            f"`lab {command} --help` must say the cap is applied against SkyPilot's catalog "
            f"estimate, not the billed rate: {text!r}"
        )


class TestTheMcpDescriptionsTellTheTruthToo:
    """MCP is the agent-facing surface, and CLAUDE.md calls it the primary one.

    The CLI guard above walks typer `OptionInfo` defaults, so it structurally cannot see MCP tool
    descriptions — which kept promising "a hard $/hr ceiling on compute enforced by the optimizer"
    after every CLI string had been corrected.
    """

    def _mcp_source(self) -> str:
        from pathlib import Path

        import lab.mcp_server as m

        return Path(m.__file__).read_text()

    def test_no_mcp_tool_promises_a_hard_ceiling(self) -> None:
        src = self._mcp_source()
        assert "hard $/hr ceiling" not in src, (
            "an MCP tool description still promises a ceiling SkyPilot cannot hold on Vast"
        )

    def test_price_cap_strict_is_discoverable(self) -> None:
        """A parameter an agent cannot read about is a parameter it will never use."""
        assert self._mcp_source().count("price_cap_strict=True") >= 2


class TestExceedsCap:
    def test_the_real_overrun_is_caught(self) -> None:
        """20260823-101602-f9e849: $2.220/hr against a $0.85 cap."""
        assert exceeds_cap(2.220, 0.85) is True

    def test_a_job_inside_the_cap_is_not(self) -> None:
        assert exceeds_cap(0.736, 0.85) is False

    def test_no_cap_means_nothing_to_exceed(self) -> None:
        assert exceeds_cap(2.220, None) is False

    def test_an_unknown_price_never_alarms(self) -> None:
        """A price we could not read is not evidence of an overrun (only definitive negatives)."""
        assert exceeds_cap(None, 0.85) is False

    def test_a_hair_over_is_tolerated(self) -> None:
        """Rounding must not manufacture a money alarm; the band is explicit."""
        assert exceeds_cap(0.85 * (1 + OVER_CAP_TOLERANCE / 2), 0.85) is False
        assert exceeds_cap(0.85 * (1 + OVER_CAP_TOLERANCE * 2), 0.85) is True


class TestMessages:
    def test_the_warning_quantifies_the_damage(self) -> None:
        msg = over_cap_warning(2.220, 0.85, "lab-x-20260823-101602-f9e849", 3 * 3600)
        assert "2.22" in msg and "0.85" in msg
        assert "2.6" in msg, "the ratio makes the size obvious at a glance"
        assert "6.66" in msg, "projected spend at the job's own timeout"
        assert "lab cancel" in msg, "a warning with no action is noise"

    def test_the_warning_survives_an_unknown_timeout(self) -> None:
        assert "2.22" in over_cap_warning(2.220, 0.85, "lab-x", None)

    def test_the_admission_error_names_the_cheapest_real_offer(self) -> None:
        msg = cap_admission_error(1.10, 0.85, "RTX4090:1")
        assert "1.10" in msg and "0.85" in msg and "RTX4090:1" in msg
        assert "--price-cap" in msg
