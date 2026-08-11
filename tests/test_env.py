import os

import pytest

from lab.env import load_lab_env


def _write_env(tmp_path, body: str):
    (tmp_path / ".env").write_text(body)
    return tmp_path


def test_loads_values_into_environ(tmp_path, monkeypatch):
    _write_env(tmp_path, "GOOGLE_CLOUD_PROJECT=my-proj\nLAB_R2_BUCKET=lab-artifacts\n")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("LAB_R2_BUCKET", raising=False)

    names = load_lab_env(tmp_path)

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "my-proj"
    assert set(names) == {"GOOGLE_CLOUD_PROJECT", "LAB_R2_BUCKET"}


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    """A shell-exported value must beat the file — otherwise a stale committed default would
    silently override the credentials an operator set for this one command."""
    _write_env(tmp_path, "GOOGLE_CLOUD_PROJECT=from-file\n")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "from-shell")

    names = load_lab_env(tmp_path)

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "from-shell"
    assert "GOOGLE_CLOUD_PROJECT" not in names


def test_missing_dotenv_is_a_noop(tmp_path):
    assert load_lab_env(tmp_path) == []


def test_returns_names_never_values(tmp_path, monkeypatch):
    """The return value is printed/logged by callers (FR-J1) — it must not carry secrets."""
    secret = "super-secret-value"
    _write_env(tmp_path, f"LAB_R2_BUCKET={secret}\n")
    monkeypatch.delenv("LAB_R2_BUCKET", raising=False)

    assert secret not in "".join(load_lab_env(tmp_path))


def test_comments_quotes_and_export_prefix(tmp_path, monkeypatch):
    _write_env(
        tmp_path,
        '# a comment\n\nexport GOOGLE_CLOUD_PROJECT="quoted-proj"\nLAB_R2_BUCKET=\'sq\'\n',
    )
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("LAB_R2_BUCKET", raising=False)

    load_lab_env(tmp_path)

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "quoted-proj"
    assert os.environ["LAB_R2_BUCKET"] == "sq"


def test_credentials_path_resolves_relative_to_repo(tmp_path, monkeypatch):
    key = tmp_path / "sa-key.json"
    key.write_text("{}")
    _write_env(tmp_path, "GOOGLE_APPLICATION_CREDENTIALS=sa-key.json\n")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    load_lab_env(tmp_path)

    # google.auth resolves this path from the *job's* cwd, not the repo — so it must be absolute.
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key)


def test_missing_credentials_file_fails_loud(tmp_path, monkeypatch):
    """A typo'd key path must fail here, not 20 minutes later inside SkyPilot provisioning."""
    _write_env(tmp_path, "GOOGLE_APPLICATION_CREDENTIALS=nope/sa-key.json\n")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(FileNotFoundError) as exc:
        load_lab_env(tmp_path)

    assert "GOOGLE_APPLICATION_CREDENTIALS" in str(exc.value)


def test_blank_credentials_is_ignored(tmp_path, monkeypatch):
    """An unfilled placeholder in .env must fall through to gcloud user ADC, not crash."""
    _write_env(tmp_path, "GOOGLE_APPLICATION_CREDENTIALS=\nGOOGLE_CLOUD_PROJECT=p\n")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    load_lab_env(tmp_path)

    assert not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "p"
