"""Laboratory — Remote Experiment Runner.

A thin control plane (CLI + MCP) over pluggable execution backends. See
``research/10-architecture.md`` for the design and ``LAB-REQUIREMENTS.md`` for the spec.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # Read the single source of truth (pyproject) rather than restating it here — this string
    # sat at "0.1.0" through the 0.2.x releases because a hand-maintained copy has nothing
    # forcing it to keep up.
    __version__ = version("laboratory")
except PackageNotFoundError:  # not installed (e.g. running from a source tree without an install)
    __version__ = "0.0.0+unknown"
