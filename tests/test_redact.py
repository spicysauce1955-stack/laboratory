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


def test_redact_leaves_plain_text_untouched():
    line = "[lab] provisioning host lab-abc-123 (RTX4090:1)"
    assert redact(line) == line


def test_redact_is_idempotent():
    once = redact(f"?api_key={SECRET}")
    assert redact(once) == once


import os

from lab.redact import install_log_redaction


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
