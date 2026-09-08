"""The sanitizer is the FR-J1 gate: nothing reaches the ledger without passing through it."""

from __future__ import annotations

import pytest

from lab.events.sanitize import MASK, ObjectKey, mask_text, sanitize_argv, sanitize_params

SECRET = "abcd1234efgh5678ijkl9012mnop3456qrst"

#: A real R2 queue key, of the shape `queue.manifest_corrupt` exists to name.
REAL_KEY = "queue/jobs/j-20260908-1a2b3c4d-shard-07-of-32.json"


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


def test_mask_text_does_not_corrupt_prose_that_merely_mentions_a_flag_name() -> None:
    """Bug 2: `_INLINE_FLAG` used to mask any `--flag value` pair whenever the flag name matched
    `_SECRET_KEY` at all — including "auth" as a bare substring of an unrelated flag ("basic-
    auth") or "key" glued inside one unrelated English word ("keyword") — deleting the next word
    of ordinary prose even though nothing secret-shaped is present."""
    assert (
        mask_text("documented the --basic-auth flag; users still hit 401s")
        == "documented the --basic-auth flag; users still hit 401s"
    )
    assert (
        mask_text("we need a --keyword search here")
        == "we need a --keyword search here"
    )
    assert (
        mask_text("run with --price-cap 1.40 and see what happens")
        == "run with --price-cap 1.40 and see what happens"
    )


def test_mask_text_still_masks_real_secrets_after_the_flag_fix() -> None:
    """The over-masking fix must not weaken real detection: a secret-shaped value is still fully
    masked whether it rides on a sensitive-named flag (bare or `--flag=`) or stands alone."""
    # sensitive flag name alone is still enough when the value is short/low-entropy
    out = mask_text("failed with --api-key=sk-live-abcdef1234567890 in the command")
    assert "sk-live-abcdef1234567890" not in out
    assert "--api-key=" in out
    # a genuinely secret-shaped value is caught even behind an unremarkable flag name
    out2 = mask_text(f"retry with --seed {SECRET} next time")
    assert SECRET not in out2
    assert MASK in out2
    # bare secret, no flag prefix at all, is still caught by the free-standing pass
    out3 = mask_text(f"got token {SECRET} from the response")
    assert SECRET not in out3
    assert MASK in out3


def test_mask_text_masks_a_flag_value_followed_by_a_boolean_flag() -> None:
    """Bug 1 (secret-leak): the bare-token value alternative used to have no restriction on its
    own shape, so it would greedily consume a *following* `--flag` as if it were the current
    flag's value (`--dry-run --api-key ...` matched flag=`--dry-run`, value=`--api-key`,
    swallowing both). That left `--api-key` consumed as somebody else's value, never itself
    tried as a flag, so its real value never got independent consideration. Confirmed live:
    this exact string used to come back completely unchanged, no masking at all."""
    out = mask_text("ran with --dry-run --api-key sk-live-abcdef1234567890 next")
    assert "sk-live-abcdef1234567890" not in out
    assert MASK in out
    assert "--api-key" in out
    assert "--dry-run" in out  # the boolean flag itself survives untouched


def test_mask_text_masks_two_secrets_separated_by_an_intervening_boolean_flag() -> None:
    """Adversarial variation on bug 1: a boolean flag sitting *between* two real flag+secret
    pairs must not eat either secret's flag."""
    out = mask_text(
        "ran with --api-key sk-live-abcdef1234567890 --verbose --token tok-zzzzzzzzzzzzzzzz done"
    )
    assert "sk-live-abcdef1234567890" not in out
    assert "tok-zzzzzzzzzzzzzzzz" not in out
    assert out.count(MASK) == 2
    assert "--verbose" in out
    assert "done" in out


def test_mask_text_does_not_span_past_a_stray_apostrophe_in_later_prose() -> None:
    """Bug 2 (corruption): the quoted-value alternative used to allow single quotes as a
    delimiter with no bound on distance — an opening `'` with no genuine closing partner nearby
    would backtrack all the way to the next literal `'` anywhere later in the string, even one
    that's just part of an ordinary contraction, silently deleting real prose in between.
    Confirmed live: this exact string used to come back with everything from the opening quote
    to the apostrophe in "it's" replaced by the mask. The secret must still be masked, but the
    unrelated tail must survive verbatim."""
    out = mask_text("run with --api-key 'sk-live-abcdef1234567890 and later it's done")
    assert "sk-live-abcdef1234567890" not in out
    assert MASK in out
    assert "and later it's done" in out


def test_mask_text_leaves_an_apostrophe_contraction_right_after_a_flag_alone() -> None:
    """A bare, non-secret, non-sensitive-flag value that happens to be an ordinary contraction
    must not be mistaken for an opening quote or otherwise mangled — it's just a bare token."""
    assert (
        mask_text("run with --reason it's broken") == "run with --reason it's broken"
    )


def test_mask_text_bounds_a_double_quoted_value_even_when_unterminated() -> None:
    """The double-quoted alternative is still explicitly length-bounded, so an unterminated `"`
    can't reach an unrelated `"` much later in a long note and swallow everything between."""
    tail = "word " * 100 + 'and she said "looks fine" at the end'
    out = mask_text(f'noted --api-key "sk-live-unterminated {tail}')
    assert "at the end" in out  # the far-away unrelated closing quote's context survives
    assert MASK in out


def test_mask_text_catches_a_real_secret_bare_and_inside_a_flag() -> None:
    """A genuinely secret-shaped value must be caught whether it stands alone in free text or
    is passed as a `--flag=value`."""
    bare = mask_text(f"got token {SECRET} from the response")
    assert SECRET not in bare
    assert MASK in bare

    flagged = mask_text(f"ran with --api-key={SECRET} set")
    assert SECRET not in flagged
    assert "--api-key=" in flagged


# --- ObjectKey: a pre-vetted object-store path an internal diagnostic may carry ----------------
#
# The bug: `r2queue.list_mirrored`/`list_entries` skip an unparseable blob and record
# `events.note("queue.manifest_corrupt", key=..., ...)`, and that note has never once named the
# object it is about — `key` matches `_SECRET_KEY`, and a real key is long and high-entropy
# enough that renaming the field doesn't rescue it either. The one identifier the diagnostic
# exists to carry was the one thing destroyed.


def test_a_plain_object_key_string_is_still_masked_both_ways() -> None:
    """The collision this fixes is real and stays real for unwrapped strings: nothing about the
    deny-list's default behaviour is relaxed. Under the field name `key` it is masked by name;
    under an innocent name it is masked by entropy."""
    assert sanitize_params({"key": REAL_KEY}) == {"key": MASK}
    assert sanitize_params({"blob": REAL_KEY}) == {"blob": MASK}


def test_a_vetted_object_key_survives_verbatim_under_a_secret_shaped_field_name() -> None:
    """`ObjectKey` is a statement about the *value*, made in lab's own source, so it also
    overrides the field-name rule — `key=` is the natural name for an object key and the whole
    point is that the diagnostic names its subject."""
    assert sanitize_params({"key": ObjectKey(REAL_KEY)}) == {"key": REAL_KEY}
    assert sanitize_params({"object_key": ObjectKey("queue/entries/reg-partial.json")}) == {
        "object_key": "queue/entries/reg-partial.json"
    }
    # nested, and inside a list, since notes are arbitrary mappings
    assert sanitize_params({"d": {"key": ObjectKey(REAL_KEY)}}) == {"d": {"key": REAL_KEY}}
    assert sanitize_params({"skipped": [ObjectKey(REAL_KEY)]}) == {"skipped": [REAL_KEY]}
    # ...but a *container* under a secret-shaped field name is still masked whole, unchanged:
    # the name rule short-circuits before recursing, and widening that would be a real
    # weakening (`{"api_keys": [<short secret>, ...]}` relies on it).
    assert sanitize_params({"keys": [ObjectKey(REAL_KEY)]}) == {"keys": MASK}


def test_a_vetted_key_is_a_plain_str_in_the_output() -> None:
    """The ledger is JSON: the record must not carry a str *subclass* out of the sanitizer."""
    out = sanitize_params({"key": ObjectKey(REAL_KEY)})["key"]
    assert type(out) is str


@pytest.mark.parametrize(
    ("name", "value"),
    [
        # An AWS-style secret access key is 40 base64 chars and genuinely contains `/`, so it is
        # *path-shaped*. It must not survive the wrapper.
        ("aws_secret_shaped", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        # ...nor when it is hiding as one segment of an otherwise plausible path.
        ("blob_segment", "queue/wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY/manifest.json"),
        ("hex_token_segment", "queue/" + "a1b2c3d4" * 8 + "/manifest.json"),
        ("oauth_token", "ya29.a0AfH6SMBx/rest"),
        ("pem", "-----BEGIN RSA PRIVATE KEY-----/x"),
        ("high_entropy_no_slash", SECRET),  # a bare token is not a path at all
        ("base64_padded", "cXVldWUvam9icy9hYmNkZWZnaGlqa2xtbm9w=="),  # `=` isn't in a key
        ("query_string", "queue/jobs/x.json?api_key=abcdef0123456789abcdef0123456789"),
        ("url_with_userinfo", "https://user:hunter2@example.com/queue/x.json"),
        ("whitespace", "queue/jobs/x.json --api-key sk-live-abcdef1234567890"),
    ],
)
def test_wrapping_a_credential_shape_does_not_exempt_it(name: str, value: str) -> None:
    """Defence in depth behind the type: even at a call site that (wrongly) wraps it, a value
    that isn't unambiguously an object key falls back to the ordinary deny-list — byte for byte
    what the same plain string gets today. A mistaken wrap degrades to the status quo; it can
    never *unmask* something."""
    assert sanitize_params({"x": ObjectKey(value)}) == sanitize_params({"x": value})


def test_wrapping_a_credential_shape_still_masks_the_credential() -> None:
    """The half of the case above that matters most, asserted directly rather than by equality:
    these are masked, not merely 'treated the same'."""
    for value in [
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "queue/wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY/manifest.json",
        "ya29.a0AfH6SMBx/rest",
        SECRET,
    ]:
        assert sanitize_params({"x": ObjectKey(value)}) == {"x": MASK}


def test_an_over_long_vetted_key_is_not_exempt() -> None:
    """A bounded exemption: an object key is a short identifier, and an unbounded pass-through
    would be a channel for pasting anything at all into the ledger. Over the bound the value
    falls back to the deny-list, which masks this one on entropy."""
    parts = [f"j-2026090{i}-1a2b3c4d-shard-0{i}-of-32.json" for i in range(9)]
    long_key = "/".join(["queue", *parts])
    assert len(long_key) > 256
    assert sanitize_params({"blob": ObjectKey(long_key)}) == {"blob": MASK}


def test_the_wrapper_cannot_be_produced_by_deserialization() -> None:
    """The load-bearing property of the mechanism: `ObjectKey` is a Python type, so no value
    crossing a trust boundary (argv, an MCP tool argument, a JSON record, a note body) can ever
    arrive as one. `json.loads` produces `str`, and a `str` takes the ordinary path."""
    import json

    decoded = json.loads(json.dumps({"key": REAL_KEY}))
    assert type(decoded["key"]) is str
    assert sanitize_params(decoded) == {"key": MASK}


def test_argv_and_note_text_are_untouched_by_the_wrapper() -> None:
    """The exemption lives only in the params walk. The argv and free-text paths — the two that
    handle user input — are not given a bypass at all."""
    assert sanitize_argv(["lab", "queue", "list", REAL_KEY]) == [
        "lab", "queue", "list", MASK,
    ]
    assert mask_text(f"failed on {REAL_KEY}") == f"failed on {MASK}"
