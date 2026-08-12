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
