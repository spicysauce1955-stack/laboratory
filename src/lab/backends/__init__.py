"""Pluggable execution backends (Design principle 6; architecture §10).

- ``local``    — subprocess on this machine (NFR-4 fallback, P0 dev loop).
- ``skypilot`` — managed remote jobs with auto-teardown + spot (P0-remote / P1).
"""
