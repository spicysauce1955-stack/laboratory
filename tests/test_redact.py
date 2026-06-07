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
