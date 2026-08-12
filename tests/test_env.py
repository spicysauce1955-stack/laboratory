import os
from pathlib import Path

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


def test_cli_env_discovery_honours_lab_repo_dir(tmp_path, monkeypatch):
    """GCP-CREDS-2: the Typer callback resolved `.env` from cwd, while every other repo-rooted
    path in the CLI goes through `_repo()` and honours `LAB_REPO_DIR`. The scheduler host is the
    documented user of that override *and* the host that most needs a service-account key: a
    systemd unit whose WorkingDirectory isn't the repo silently loaded no `.env` at all, and the
    failure surfaced one layer down as an opaque auth error at 3am."""
    from lab.cli import _load_env

    _write_env(tmp_path, "GOOGLE_CLOUD_PROJECT=from-lab-repo-dir\n")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)  # a cwd that is not the repo

    _load_env()

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "from-lab-repo-dir"


def test_dotenv_is_excluded_from_the_workdir_sync():
    """GCP-CREDS-4. `build_task(..., workdir=Path.cwd())` rsyncs the repo root to the remote.
    `.env` was *believed* excluded because it is git-ignored — but SkyPilot's `get_excluded_files`
    uses `.skyignore` **instead of** `.gitignore` when one exists, and this repo commits a
    `.skyignore`. So the git-ignore was never consulted and `.env` shipped to every box.

    Today `.env` holds only paths, so the blast radius is a disclosed filesystem layout rather
    than a key — but it is precisely the file a user pastes an R2 secret into, and LAB-BUGS §7 is
    this repo's history of a secret reaching a persisted artifact. Asserted against SkyPilot's own
    exclusion logic, so it tracks what really syncs rather than what we intended.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from sky.data.storage_utils import get_excluded_files

    repo = Path(__file__).resolve().parents[1]

    # `get_excluded_files` returns files that EXIST and match, not the patterns themselves — and
    # `.env` is git-ignored, so it exists on a developer's box and never on a fresh checkout.
    # Asserting against the repo directly therefore passed locally and failed on CI, testing the
    # machine rather than the code. Copy the real `.skyignore` next to a real `.env` instead: same
    # question, same vendor function, no dependency on who is running it.
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(td)
        shutil.copy2(repo / ".skyignore", fixture / ".skyignore")
        (fixture / ".env").write_text("LAB_R2_BUCKET=secret-looking-value\n")
        (fixture / "keep_me.py").write_text("# a file that must still sync\n")

        excluded = {e.strip("/") for e in get_excluded_files(str(fixture))}

    assert ".env" in excluded
    assert "keep_me.py" not in excluded  # the pattern excludes .env, not everything


# --- LAB_REPO_DIR: one resolution, honoured everywhere ----------------------------------------


def test_repo_root_honours_lab_repo_dir(tmp_path, monkeypatch):
    """The override belongs in `repo_root` itself. It used to live in `cli._repo()`, which library
    code cannot reach — so `default_lab`, `default_queue` and the MCP entrypoint all silently fell
    back to cwd while a handful of CLI commands honoured the variable."""
    from lab.manifest import repo_root

    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)

    assert repo_root() == tmp_path


def test_an_explicit_start_is_never_overridden_by_the_env(tmp_path, monkeypatch):
    """`repo_root(start)` asks about a *specific* directory — provenance capture uses it to find
    the work tree containing a known path. An ambient override must not hijack that answer, or the
    commit recorded on a manifest stops describing the code that ran."""
    from lab.manifest import repo_root

    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path / "elsewhere"))
    repo = Path(__file__).resolve().parents[1]

    assert repo_root(repo) == repo


def test_blank_lab_repo_dir_falls_through_to_cwd(tmp_path, monkeypatch):
    """An unfilled `.env.example` placeholder must behave as unset, like every other setting."""
    from lab.manifest import repo_root

    monkeypatch.setenv("LAB_REPO_DIR", "  ")
    monkeypatch.chdir(tmp_path)

    assert repo_root() == tmp_path


def test_default_lab_is_rooted_at_lab_repo_dir(tmp_path, monkeypatch):
    """The split-brain this closes. `default_lab` is the shared constructor for both the CLI and
    the MCP server, and it decides *both* the repo used for provenance and where `runs/` lives —
    so on a host with `LAB_REPO_DIR` set it read a different `runs/` than the scheduler wrote."""
    from lab.core import default_lab

    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)

    lab = default_lab()

    assert lab.repo == tmp_path
    assert lab.home == tmp_path / "runs"


def test_default_queue_is_rooted_at_lab_repo_dir(tmp_path, monkeypatch):
    """The repo-local queue is what `lab register` writes and `lab scheduler tick` reads. If they
    disagree about the repo root, jobs queue into a directory the scheduler never looks at."""
    from lab.scheduler.queue import default_queue

    monkeypatch.delenv("LAB_QUEUE_DIR", raising=False)
    monkeypatch.delenv("LAB_R2_ENDPOINT", raising=False)
    monkeypatch.delenv("LAB_R2_BUCKET", raising=False)
    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)

    assert default_queue().root == tmp_path / "queue"


def test_lab_repo_dir_resolves_to_the_work_tree_root(monkeypatch):
    """Review finding: the override returned its raw value, skipping the git lookup — so a value
    pointing *inside* a repo was handed to `current_commit`/`is_dirty`/`capture_diff`, which all
    assume a work-tree root and pin provenance from it."""
    from lab.manifest import repo_root

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LAB_REPO_DIR", str(repo / "tests"))

    assert repo_root() == repo


def test_lab_repo_dir_expands_a_tilde(monkeypatch):
    """`.env` is hand-edited, so `~/laboratory` is the natural thing to write."""
    from lab.manifest import repo_root

    monkeypatch.setenv("LAB_REPO_DIR", "~")
    assert repo_root() == Path.home().resolve()


def test_a_non_repo_lab_repo_dir_still_works(monkeypatch, tmp_path):
    """The scheduler host is allowed to point at a plain directory — it must not hard-fail."""
    from lab.manifest import repo_root

    plain = tmp_path / "notarepo"
    plain.mkdir()
    monkeypatch.setenv("LAB_REPO_DIR", str(plain))

    assert repo_root() == plain.resolve()


def test_lab_repo_dir_warns_when_it_shadows_the_tree_you_are_in(tmp_path, monkeypatch, capsys):
    """Review finding: the override governs `Lab.repo` — the tree whose commit is pinned and
    whose contents are uploaded. A laptop with it set in `.env` that then runs from a second
    checkout or a git worktree would launch the *other* tree's code and record a commit that
    never contained the change, with no error anywhere."""
    from lab.cli import _warn_if_repo_override_shadows_cwd

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(repo)  # standing in a real work tree that is NOT the override

    _warn_if_repo_override_shadows_cwd()

    assert "LAB_REPO_DIR" in capsys.readouterr().err


def test_no_warning_on_a_host_whose_cwd_is_not_a_repo(tmp_path, monkeypatch, capsys):
    """The scheduler host's intended use: WorkingDirectory isn't a work tree, so there is no
    shadowing and no warning to cry wolf with every tick."""
    from lab.cli import _warn_if_repo_override_shadows_cwd

    elsewhere = tmp_path / "notarepo"
    elsewhere.mkdir()
    monkeypatch.setenv("LAB_REPO_DIR", str(tmp_path))
    monkeypatch.chdir(elsewhere)

    _warn_if_repo_override_shadows_cwd()

    assert capsys.readouterr().err == ""
