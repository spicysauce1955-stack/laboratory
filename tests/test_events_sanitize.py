"""The sanitizer is the FR-J1 gate: nothing reaches the ledger without passing through it."""

from __future__ import annotations

import pytest

from lab.events.sanitize import MASK, mask_text, sanitize_argv, sanitize_params

SECRET = "abcd1234efgh5678ijkl9012mnop3456qrst"


@pytest.mark.parametrize(
    "key",
    ["api_key", "vast_api_key", "token", "access_token", "client_secret", "password",
     "credential", "AUTH_HEADER"],
)
def test_secret_shaped_keys_are_masked(key: str) -> None:
    assert sanitize_params({key: SECRET}) == {key: MASK}


def test_ordinary_params_survive_verbatim() -> None:
    params = {"command": "python experiments/x.py", "backend": "cpu", "cloud": "gcp", "seeds": "0-31"}
    assert sanitize_params(params) == params


def test_secret_shaped_values_are_masked_under_innocent_keys() -> None:
    out = sanitize_params({"note": "ya29.a0AfH6SM", "pem": "-----BEGIN RSA PRIVATE KEY-----x"})
    assert out == {"note": MASK, "pem": MASK}


def test_high_entropy_value_is_masked_but_ids_and_shas_are_not() -> None:
    assert sanitize_params({"x": SECRET}) == {"x": MASK}
    # hex ids (commits, cell ids, job ids) are not credential-shaped and stay readable
    assert sanitize_params({"commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"})["commit"].startswith("a1b2")
    assert sanitize_params({"job_id": "j-4f2a8c1e"}) == {"job_id": "j-4f2a8c1e"}


def test_long_strings_and_lists_are_truncated() -> None:
    out = sanitize_params({"blob": "a b " * 400, "many": list(range(100))})
    assert len(out["blob"]) <= 512 + 1  # + the ellipsis marker
    assert out["blob"].endswith("…")
    assert len(out["many"]) == 33  # 32 items + the "…N more" marker
    assert out["many"][-1] == "…68 more"


def test_nested_structures_are_sanitized() -> None:
    out = sanitize_params({"resources": {"cpus": 4, "api_key": SECRET}})
    assert out == {"resources": {"cpus": 4, "api_key": MASK}}


def test_argv_masks_the_value_after_a_secret_flag() -> None:
    argv = ["lab", "submit", "--vast-api-key", SECRET, "--backend", "cpu"]
    assert sanitize_argv(argv) == ["lab", "submit", "--vast-api-key", MASK, "--backend", "cpu"]


def test_argv_masks_inline_secret_flags() -> None:
    assert sanitize_argv([f"--token={SECRET}"]) == [f"--token={MASK}"]


def test_unserializable_values_degrade_to_a_type_name() -> None:
    assert sanitize_params({"obj": object()})["obj"].startswith("<object")


def test_long_hex_secret_is_masked_but_short_hex_ids_are_not() -> None:
    # 64-char hex (typical HMAC token length) under innocent key → masked by base64 pattern
    # (64 chars of [0-9a-f] also matches [A-Za-z0-9+/]{40,})
    long_hex_repeated = "a" * 64
    assert sanitize_params({"signature": long_hex_repeated}) == {"signature": MASK}
    # Also test genuinely random-looking 64-char hex to exercise entropy path. The key must be
    # innocent (not "token"/"key"/etc.) — otherwise `_SECRET_KEY` masks it by name before the
    # value is ever inspected, and the assertion would pass without touching the entropy path
    # it claims to exercise (Finding 9).
    random_hex_64 = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0f1e2d3c4b5a6f7e8d9c0a1b"
    assert sanitize_params({"digest": random_hex_64}) == {"digest": MASK}
    # 40-char commit SHA → still NOT masked (within exemption)
    assert sanitize_params({"commit": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"})["commit"].startswith("a1b2")
    # 8-hex cell id → still NOT masked
    assert sanitize_params({"cell_id": "a1b2c3d4"}) == {"cell_id": "a1b2c3d4"}


def test_argv_masks_a_flag_value_hidden_inside_one_quoted_command_string() -> None:
    """The most common lab invocation puts a whole command line in one token (`lab submit -c
    "python train.py --hf-token=..."`), so the flag-aware masking above must also apply *inside*
    a single argv token, not just across separate tokens. Before the fix, the secret value has
    spaces around it, so `_looks_secret` bails (it refuses anything containing a space), and
    `redact()` doesn't know `--hf-token=`, so the whole string sails through unmasked."""
    secret = "hf_AbCdEf0123456789abcdefghij"
    argv = ["submit", "-c", f"python train.py --hf-token={secret}"]
    out = sanitize_argv(argv)
    assert secret not in out[2]
    assert "--hf-token=" in out[2]  # the value is masked, not the whole string
    assert "python train.py" in out[2]  # readability: the rest of the command survives


def test_params_masks_a_flag_value_hidden_inside_one_command_string() -> None:
    """Same bug, reached through `sanitize_params` (MCP's path) instead of `sanitize_argv`
    (the CLI's): `{'command': 'python x.py --password=hunter2'}` must not leak `hunter2`."""
    out = sanitize_params({"command": "python x.py --password=hunter2"})
    assert "hunter2" not in out["command"]
    assert "--password=" in out["command"]
    assert "python x.py" in out["command"]


def test_sanitize_argv_degrades_instead_of_raising() -> None:
    # A non-string token raises AttributeError on .startswith mid-loop. The guard must turn
    # that into a masked result, never an exception escaping into the command.
    assert sanitize_argv(["lab", "submit", 123]) == [MASK]  # type: ignore[list-item]


def test_mask_text_has_no_length_cap() -> None:
    """`mask_text` is what `lab.notes._clean` uses instead of `sanitize_argv`: a note body's
    whole purpose is holding a detailed write-up in full, so the ledger's 512-char argv/param cap
    (`test_long_strings_and_lists_are_truncated` above) must not apply to it."""
    blob = "a b " * 400  # > 512 chars, same shape sanitize_params truncates
    assert mask_text(blob) == blob


def test_mask_text_still_masks_a_secret() -> None:
    out = mask_text("failed with --api-key=sk-live-abcdef1234567890 in the command")
    assert "sk-live-abcdef1234567890" not in out
    assert "--api-key=" in out


def test_mask_text_degrades_instead_of_raising() -> None:
    assert mask_text(123) == 123  # type: ignore[arg-type]  # not a string: must not raise


def test_mask_text_masks_a_bare_secret_with_no_flag_prefix() -> None:
    """Bug 1: a secret-shaped string with nothing flag-like in front of it — the ordinary shape
    of pasted error text/tracebacks, which is exactly what `lab note` exists to hold — must still
    be masked. A regex that only recognizes `--flag value` shape misses this entirely."""
    secret = "AKIAIOSFODNN7EXAMPLEXXXXXXXXXXXXXXXXXXXXXXXXXX"
    out = mask_text(f"Authentication failed with key {secret} during connect")
    assert secret not in out
    assert MASK in out
    assert "Authentication failed with key" in out
    assert "during connect" in out


def test_mask_text_leaves_hyphenated_words_alone() -> None:
    """Bug 2: no word-boundary anchor before the flag-like token let the regex fire mid-word
    inside ordinary hyphenated compounds ending in something flag-and-secret-shaped
    (`pass-key`, `well-authenticated`), corrupting free text and swallowing the next word."""
    assert (
        mask_text("we tested the pass-key rotation before shipping")
        == "we tested the pass-key rotation before shipping"
    )
    assert (
        mask_text("well-authenticated users can proceed")
        == "well-authenticated users can proceed"
    )


def test_mask_text_masks_a_quoted_multiword_flag_value_completely() -> None:
    """Bug 3: the value-capture group used to stop at the first whitespace, so a quoted
    multi-word value after a flag was only partially masked (`--api-key "abc def ghi"` left
    `def ghi"` in the clear)."""
    out = mask_text('failed with --api-key "abc def ghi" in the command')
    assert "abc" not in out
    assert "def" not in out
    assert "ghi" not in out
    assert "--api-key" in out
    assert "in the command" in out


def test_mask_text_leaves_ordinary_prose_with_apostrophes_untouched() -> None:
    """The regression this whole rewrite exists to prevent: `shlex.split` used to treat prose
    apostrophes/quotes as shell quoting and silently mangle ordinary text. Confirm plain prose
    with contractions and hyphenated words survives completely verbatim."""
    prose = (
        "it's a re-authorized, non-secret change that wasn't flagged; "
        "she said \"looks fine\" and we're done"
    )
    assert mask_text(prose) == prose


def test_mask_text_catches_a_real_secret_bare_and_inside_a_flag() -> None:
    """A genuinely secret-shaped value must be caught whether it stands alone in free text or
    is passed as a `--flag=value`."""
    bare = mask_text(f"got token {SECRET} from the response")
    assert SECRET not in bare
    assert MASK in bare

    flagged = mask_text(f"ran with --api-key={SECRET} set")
    assert SECRET not in flagged
    assert "--api-key=" in flagged
