import pytest

from lab.redact import redact

SECRET = "0000000000000000000000000000000000000"


def test_redact_masks_api_key_query_param():
    line = f"https://console.vast.ai/api/v0/asks/33945613/?api_key={SECRET}"
    out = redact(line)
    assert SECRET not in out
    assert "api_key=" in out and "REDACTED" in out


def test_redact_masks_generic_key_params():
    assert SECRET not in redact(f"url?token_key={SECRET}&x=1")
    assert SECRET not in redact(f"url?foo=1&secret_key={SECRET}")


def test_redact_masks_authorization_header():
    assert "Bearer-xyz" not in redact("Authorization: Bearer-xyz")
    assert "REDACTED" in redact("Authorization: Bearer-xyz")


def test_redact_masks_authorization_token_after_space():
    # A real `Authorization: Bearer <token>` has a space — the token (not just the scheme)
    # must be masked, or the credential leaks.
    token = "eyJhbGciOiJIUzI1NiJ9.payload.sig"
    out = redact(f"Authorization: Bearer {token}")
    assert token not in out and "Bearer" not in out
    assert "REDACTED" in out


def test_redact_leaves_plain_text_untouched():
    line = "[lab] provisioning host lab-abc-123 (RTX4090:1)"
    assert redact(line) == line


def test_redact_is_idempotent():
    once = redact(f"?api_key={SECRET}")
    assert redact(once) == once


def test_install_log_redaction_scrubs_fd_output(tmp_path, capfd):
    log = tmp_path / "logs.txt"
    # Run in a child process: install_log_redaction reassigns fds 1/2 for the whole process,
    # which would clobber the test runner's stdout if done in-process.
    import subprocess
    import sys

    secret = "0000000000000000000000000000000000000"
    code = (
        "import os,sys; from lab.redact import install_log_redaction;"
        f"install_log_redaction({str(log)!r});"
        f"os.write(1, b'GET /asks/1/?api_key={secret}\\n');"
        "sys.stdout.flush()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    content = log.read_text()
    assert secret not in content
    assert "REDACTED" in content


GCP_TOKEN = "ya29.a0AfH6SMBx7-fake-token-value_123"


def test_redact_masks_gcp_oauth_token_in_json():
    out = redact(f'{{"access_token": "{GCP_TOKEN}", "expires_in": 3599}}')
    assert GCP_TOKEN not in out
    assert '"access_token"' in out and "REDACTED" in out


def test_redact_masks_gcp_refresh_token_and_private_key_fields():
    assert "1//fake-refresh" not in redact('{"refresh_token": "1//fake-refresh"}')
    key_material = "-----BEGIN PRIVATE KEY-----\\nMIIfake\\n-----END PRIVATE KEY-----\\n"
    assert "MIIfake" not in redact(f'{{"private_key": "{key_material}"}}')
    assert "sekret" not in redact('{"client_secret": "sekret"}')


def test_redact_masks_bare_ya29_token_anywhere():
    # gcloud/SkyPilot log OAuth tokens outside JSON too, e.g. in curl commands/URLs.
    out = redact(f"curl -H 'X-Auth: {GCP_TOKEN}' https://compute.googleapis.com/")
    assert GCP_TOKEN not in out and "ya29." in out and "REDACTED" in out


def test_redact_gcp_patterns_are_idempotent():
    once = redact(f'{{"access_token": "{GCP_TOKEN}"}} and bare {GCP_TOKEN}')
    assert redact(once) == once


# --- GCP-CREDS-5: GCS signed-URL credentials -------------------------------------------------

SIGNED_URL = (
    "https://storage.googleapis.com/skypilot-filemounts-lab-abc123/workdir.zip"
    "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
    "&X-Goog-Credential=lab-sa%40my-project.iam.gserviceaccount.com%2F20260812%2Fauto%2Fstorage"
    "%2Fgoog4_request"
    "&X-Goog-Date=20260812T143000Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host"
    "&X-Goog-Signature=7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"
)


def test_redact_masks_the_gcs_signed_url_signature():
    """The signature IS the credential: anyone holding this URL can read the object until it
    expires. SkyPilot's bucket staging emits these into logs, which stream to `logs.txt` and R2."""
    out = redact(SIGNED_URL)
    assert "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069" not in out
    assert "REDACTED" in out


def test_redact_masks_the_gcs_signed_url_credential():
    """`X-Goog-Credential` carries the signing service account and its scope — not secret on its
    own, but it names the identity to attack and pairs with the signature."""
    out = redact(SIGNED_URL)
    assert "lab-sa%40my-project.iam.gserviceaccount.com" not in out


def test_redact_keeps_the_signed_url_readable():
    """Masking must leave the line diagnosable — the bucket and object still identify what was
    being staged, which is the whole reason the log line is useful."""
    out = redact(SIGNED_URL)
    assert "skypilot-filemounts-lab-abc123/workdir.zip" in out
    assert "X-Goog-Signature=" in out


def test_redact_signed_url_patterns_are_idempotent():
    once = redact(SIGNED_URL)
    assert redact(once) == once


# --- S3/R2 presigned URLs and token-bearing query params ----------------------------------------

R2_PRESIGNED = (
    "GET https://acc.r2.cloudflarestorage.com/lab-artifacts/j-1/results.csv"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=8c1f3a9e2b7d4c6f0a5e9b3d1c7f2a4e%2F20261119%2Fauto%2Fs3%2Faws4_request"
    "&X-Amz-Security-Token=FwoGZXIvYXdzEJr"
    "&X-Amz-Signature=" + "ab" * 32
)


def test_redact_masks_an_s3_presigned_url_but_keeps_it_readable():
    """The R2 twin of the GCS signed URL above: the signature grants access, the credential names
    the access key id, and a session token is a credential outright."""
    out = redact(R2_PRESIGNED)
    assert "ab" * 32 not in out
    assert "8c1f3a9e2b7d4c6f0a5e9b3d1c7f2a4e" not in out
    assert "FwoGZXIvYXdzEJr" not in out
    assert "lab-artifacts/j-1/results.csv" in out and "X-Amz-Signature=" in out
    assert redact(out) == out


def test_redact_masks_token_and_password_query_params():
    out = redact("https://api.x.io/v1?token=abc123&access_token=zzz&password=pw&sig=s1&page=2")
    for secret in ("abc123", "=zzz", "=pw&", "s1"):
        assert secret not in out, out
    assert "page=2" in out
    assert redact(out) == out


def test_redact_masks_password_and_secret_json_fields():
    out = redact('{"user": "x", "password": "hunter2", "secret": "s3", "api_key": "k"}')
    assert "hunter2" not in out and '"s3"' not in out and '"k"' not in out
    assert '"user": "x"' in out


# --- credential shapes that need no name in front of them (stress run 2026-11-19) -------------
#
# `lab export --logs` writes a bundle meant to be committed, and its "redacted" logs.txt carried
# every one of these verbatim: `redact` only knew the cloud providers' query-param and header
# shapes. Each fake is built to the real format's length and alphabet; the `FAKE…` variants are
# the stress run's own (`experiments/leaky.py`), which are all letters and so slip past any
# entropy bar — the prefix has to be enough.

B62 = "aZ3kQ9xW2mP7vB4nR8tY1cL6jH5gF0dSe2Wq7Lp4Mn8Bv3Cx6Zk"  # base62, deterministic
TOKENS = {
    "github classic": "ghp_" + B62[:36],
    "github oauth": "gho_" + B62[1:37],
    "github fake": "ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE",
    "github fine-grained": "github_pat_11ABCDEFG0" + B62[:12] + "_" + (B62 * 2)[:59],
    "openai legacy": "sk-" + B62[:48],
    "openai fake": "sk-FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE",
    "openai project": "sk-proj-" + (B62 * 3)[:120].replace("Q", "-").replace("W", "_"),
    "anthropic": "sk-ant-api03-" + (B62 * 2)[:93] + "AA",
    "stripe": "sk_live_" + B62[:24],
    "huggingface": "hf_" + B62[:34],
    "digitalocean": "dop_v1_" + "4c6f0a5e9b3d1c7f" * 4,
    "digitalocean fake": "dop_v1_" + "FAKE" * 16,
    "slack bot": "xoxb-8906828836-8906828836-" + B62[:24],
    "slack user": "xoxp-1234567890-1234567890-1234567890-" + "ab12" * 8,
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "google api key": "AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI",
}


@pytest.mark.parametrize("name", list(TOKENS))
def test_redact_masks_a_bare_credential_shape(name: str) -> None:
    token = TOKENS[name]
    for line in (
        f"github token {token}",
        f"export X={token}",  # the name says nothing; the shape has to
        f'{{"value": "{token}"}}',
        f"curl -H 'X-Api: {token}' https://example.com",
        f"{token}",
    ):
        out = redact(line)
        assert token not in out, (name, out)
        assert "REDACTED" in out
        assert redact(out) == out  # idempotent
    # The prefix survives, so a reader can still tell *what* leaked and rotate it.
    assert redact(f"token {token} used").startswith("token ")


@pytest.mark.parametrize(
    "line",
    [
        "pip install scikit-learn sk-learn==1.5",
        "sk-learn-is-a-python-library-for-machine-learning",  # no digit, hyphenated words
        "task-0123456789abcdefghijklmnop finished",  # `sk-` inside a word
        "disk-1234567890abcdefghijklmnopq is full",
        "risk-free rate 0.04",
        "ghp_short is not a token",
        "the hf_hub cache lives in ~/.cache/huggingface",
        "xoxb is a slack prefix",
        "eyJ alone is just base64 for {\"",
        "-----BEGIN CERTIFICATE----- MIIBszCCAVmgAwIBAgIU -----END CERTIFICATE-----",
        "[lab] provisioning host lab-abc-123 (RTX4090:1)",
        "step 3 loss=0.25 lr=0.001 seed=3 alpha=0.5",
    ],
)
def test_redact_leaves_lookalikes_alone(line: str) -> None:
    """The log on disk is the only copy of the experiment's output: masking something that is
    not a credential destroys data and protects nothing."""
    assert redact(line) == line


PEM_BODY = [
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj",
    "MzEfYyjiWA4R4/M2bS1GB4t7NXp98C3SC6dVMvDuictGeurT8jNbvJZHtCSuYEvu",
    "NMoSfm76oqFvAp8Gy0iz5sxjZmSnXyCdPEovGhLa0VzMaQ8s+CLOyS56YyCFGeJZ",
    "agMOFn0=",
]


def _pem(kind: str = "PRIVATE KEY") -> str:
    return "\n".join([f"-----BEGIN {kind}-----", *PEM_BODY, f"-----END {kind}-----"])


@pytest.mark.parametrize("kind", ["PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY",
                                  "OPENSSH PRIVATE KEY", "PGP PRIVATE KEY BLOCK"])
def test_redact_masks_a_whole_multiline_private_key_block(kind: str) -> None:
    text = f"loading credentials\n{_pem(kind)}\nstep 3 loss=0.25\n"
    out = redact(text)
    for body in PEM_BODY:
        assert body not in out
    # The markers stay — a reader must be able to see that a key was printed and rotate it —
    # and so does everything around the block.
    assert f"-----BEGIN {kind}-----" in out and f"-----END {kind}-----" in out
    assert out.startswith("loading credentials\n") and out.endswith("step 3 loss=0.25\n")
    assert out.count("\n") == text.count("\n")  # line structure intact
    assert redact(out) == out


def test_redact_masks_a_private_key_printed_on_one_line() -> None:
    line = "-----BEGIN PRIVATE KEY----- MIIFAKEFAKEFAKE -----END PRIVATE KEY-----"
    out = redact(line)
    assert "MIIFAKEFAKEFAKE" not in out
    assert out.startswith("-----BEGIN PRIVATE KEY-----")
    assert out.endswith("-----END PRIVATE KEY-----")


def test_redact_masks_a_timestamped_key_block_from_an_older_log() -> None:
    """`lab export --logs` redacts logs.txt as it is on disk, and a log written before this fix
    is timestamped line by line — the body lines carry a stamp in front."""
    stamped = "\n".join(f"2026-08-21T04:12:33.{i:03d}Z {ln}" for i, ln in enumerate(
        _pem().splitlines()))
    out = redact(stamped + "\n2026-08-21T04:12:34.000Z step 3\n")
    for body in PEM_BODY:
        assert body not in out
    assert out.endswith("2026-08-21T04:12:34.000Z step 3\n")
    assert "2026-08-21T04:12:33.002Z " in out  # stamps survive, so the timeline does


def test_a_stray_begin_marker_does_not_swallow_the_rest_of_the_log() -> None:
    """A key block ends at its END marker *or* at the first line that cannot be key material.
    Masking everything after a lone BEGIN would destroy the only copy of the run's output."""
    text = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEF\nstep 3 loss=0.25 done\n"
    out = redact(text)
    assert "MIIEvQIBADANBgkqhkiG9w0BAQEF" not in out
    assert "step 3 loss=0.25 done" in out


def test_the_log_writer_masks_a_key_block_that_arrives_line_by_line() -> None:
    """The supervisor's writer redacts one line at a time, so it has to carry "inside a key
    block" from one line to the next — no single line of a key's body says what it is."""
    import io

    from lab.redact import TimestampingWriter

    sink = io.StringIO()
    w = TimestampingWriter(sink, timestamps=False)
    for ln in ["before", *_pem("RSA PRIVATE KEY").splitlines(), "after the key"]:
        w.feed((ln + "\n").encode())
    w.close()
    out = sink.getvalue()
    for body in PEM_BODY:
        assert body not in out
    assert out.splitlines()[0] == "before" and out.splitlines()[-1] == "after the key"
    assert "-----END RSA PRIVATE KEY-----" in out
