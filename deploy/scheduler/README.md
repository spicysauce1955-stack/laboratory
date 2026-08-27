# Scheduler host deployment (spec §7)

The scheduler is **stateless**: queue entries, control, bundles, and mirrored job manifests all
live in R2, so this host can be destroyed and recreated at any time.

## Redeploy (primary path, since 2026-08)

```bash
deploy/scheduler/deploy.sh vX.Y.Z
```

Builds a new droplet from the pinned tag, takes the old one out of service, then proves the new
one can actually launch a job before permanently deleting the old one — an immutable blue-green
swap, never an in-place mutation. No SSH involved. Safe to re-run: every step before the final
delete leaves the previous droplet as a fallback. Most failures roll back automatically, but a
compound failure (an inconclusive smoke test, or a rollback step itself failing) surfaces with a
manual-action message instead of self-healing.

Requires `doctl` (authenticated) and the same controller-side secrets the manual steps below
always needed: `~/.config/vastai/vast_api_key`, `~/.cloudflare/r2.credentials`,
`$LAB_R2_ENDPOINT` exported.

Full design + the two real bugs an adversarial review caught before this shipped:
`docs/superpowers/specs/2026-08-27-scheduler-deploy-cutover-design.md`.

The sections below (manual provisioning via the `playground` repo, in-place SSH upgrade) are kept
as reference/fallback — `deploy.sh` is the path to actually use.

## Provision (playground repo)

Use the playground project's `cloud-digitalocean` backend to create the smallest droplet
(the tick is tiny and I/O-bound), then run the steps below (manually or via an Ansible role in
that repo).

The host runs a **pinned release of the lab** against a clone of your **experiment project** —
the same split as on your laptop. The lab is a tool; the project supplies the git history the
scheduler pins as provenance, the `runs/` it writes, and the `.env` it reads.

There is no clone of the *lab* repo on this host, so there is no `deploy/` directory to copy from:
the unit files are fetched from the same tag the tool is pinned to. `$TAG` below is that tag —
set it once and the whole runbook is consistent.

```bash
TAG=v0.5.0
RAW=https://raw.githubusercontent.com/spicysauce1955-stack/laboratory/$TAG/deploy/scheduler
```

1. Install git, and uv **somewhere every user can run it** — the tick runs as `lab`, not root:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
   ```
2. Create the service user **before** installing anything: `useradd --create-home --system lab`.
3. Install the lab **as that user**, at the version you want the host to run:
   ```bash
   sudo -u lab -H uv tool install \
     "laboratory[skypilot,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@$TAG"
   ```
   This puts the entrypoint at **`/home/lab/.local/bin/lab`** — the path the unit must use.
   Installing as root instead puts it under `/root/.local/bin`, which is mode 0700: `User=lab`
   then cannot execute it and systemd fails the unit with **203/EXEC**.
   **Add `gcp` to the extras if you will register `--cloud gcp` jobs** — without it the host
   cannot provision on GCP at all, and the failure only shows up at launch time, unattended, at
   3am.
4. Clone the experiment project as `lab`:
   ```bash
   install -d -o lab -g lab /opt/tempotron-capacity
   sudo -u lab -H git clone <project remote> /opt/tempotron-capacity
   ```
5. Fetch the env template, fill in real credentials (mode 0600, owner `lab`):
   ```bash
   install -d -m 0755 /etc/lab
   curl -fsSL $RAW/scheduler.env.example -o /etc/lab/scheduler.env
   chown lab:lab /etc/lab/scheduler.env && chmod 600 /etc/lab/scheduler.env
   ```
6. Fetch the unit + timer, then point the unit at the project and the right binary — three lines:
   ```bash
   curl -fsSL $RAW/lab-scheduler.service -o /etc/systemd/system/lab-scheduler.service
   curl -fsSL $RAW/lab-scheduler.timer   -o /etc/systemd/system/lab-scheduler.timer
   ```
   ```ini
   WorkingDirectory=/opt/tempotron-capacity
   Environment=LAB_REPO_DIR=/opt/tempotron-capacity
   ExecStart=/home/lab/.local/bin/lab scheduler tick --backend skypilot
   ```
   As of v0.5.0 the checked-in `ExecStart`/`WorkingDirectory`/`Environment` already point at the
   right paths for a `tempotron-capacity` deploy — no substitution needed. (This used to require a
   manual patch; it doesn't anymore. If you're deploying against a *different* experiment project,
   you still need to edit those three lines by hand.)
7. `systemctl daemon-reload && systemctl enable --now lab-scheduler.timer`

### Upgrading the host

Re-install pinned to the new tag. `uv tool upgrade laboratory` is effectively a no-op here — the
requirement is a git tag, so there is nothing newer for it to resolve to:

```bash
sudo -u lab -H uv tool install --force \
  "laboratory[skypilot,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.6.0"
sudo -u lab -H /home/lab/.local/bin/lab --version   # confirm the new version is live
```

Keep the extras identical to the install line — `--force` replaces the environment wholesale, so
an extra dropped here is a backend silently gone at 3am. Nothing else moves: the project clone and
`/etc/lab/scheduler.env` are untouched.

### Cutting over from a `/opt/laboratory` clone

Hosts provisioned before v0.5.0 ran `uv run lab scheduler tick` inside a clone of the *lab* repo.
To move to the layout above:

1. **Drain the queue first** — `lab queue list` must be empty, and no job in flight. The queue
   dir and `runs/` are resolved relative to the repo, so pending registrations under
   `/opt/laboratory` are stranded the moment `LAB_REPO_DIR` moves.
2. `systemctl stop lab-scheduler.timer`, then follow steps 2–7 above.
3. Verify with one cheap registered job end to end before trusting it with a night's work.

## Google Cloud credentials (only for `--cloud gcp` registrations)

The host needs its own credentials — it is a different machine from your laptop, and nothing is
copied to it. Skip this section entirely if you only use Vast/DO.

```bash
# as root, with a service-account key created per docs/guides/gcp-backend.md
install -o lab -g lab -m 600 lab-sa.json /home/lab/.config/gcloud/lab-sa.json
# ALSO symlink it to the well-known ADC path — see the warning below
sudo -u lab ln -s /home/lab/.config/gcloud/lab-sa.json \
     /home/lab/.config/gcloud/application_default_credentials.json
```

then set `GOOGLE_APPLICATION_CREDENTIALS` (the **path**) and `GOOGLE_CLOUD_PROJECT` in
`/etc/lab/scheduler.env`.

> **The symlink is not optional.** `EnvironmentFile` sets the variables for the *tick* process,
> but SkyPilot runs a long-lived **API-server daemon** that does not inherit them — a daemon
> started by an earlier tick keeps whatever environment it was born with. The well-known ADC path
> is read by every process regardless. Same gotcha as on the laptop; see the *Credentials*
> section of `docs/guides/gcp-backend.md`.

The service account needs the six roles listed in that guide, and the project needs both
`compute.googleapis.com` and `cloudresourcemanager.googleapis.com` enabled.

## Verify

- `sudo -u lab -H /home/lab/.local/bin/lab --version` — the tool is installed, executable **by the
  user the unit runs as**, and at the tag you meant to pin. This is the check that catches
  203/EXEC before systemd does.
- `systemctl list-timers lab-scheduler.timer` — next tick scheduled.
- From the laptop: `lab queue list` — `heartbeat_age_s` under ~120.
- If you configured GCP: see the section below. Do it *before* registering a deferred GCP job —
  a credential problem discovered at launch time wastes the whole night the job was scheduled for.

## GCP host check (`GCP-CREDS-1`) — ~20 minutes, do it once

The runbook above has never been confirmed on the live droplet. The failure mode it guards
against is a deferred GCP job that queues fine, passes its triggers, and dies at launch at 3am
unattended — which is the entire point of the feature.

`lab doctor --cloud gcp` did not exist when this gap was written; it now answers the whole
question in one shot, including whether SkyPilot's daemon agrees (the thing the symlink above
exists for). **On the host, as the `lab` user:**

```bash
cd /opt/tempotron-capacity
sudo -u lab env $(grep -v '^#' /etc/lab/scheduler.env | xargs) \
  /home/lab/.local/bin/lab doctor --cloud gcp   # credentials, project, billing, APIs, IAM, quota
```

Every check should be `ok` or `skip`; **only a definitive negative blocks**, so a `skip` is not a
failure. If it reports a credential problem, re-check the ADC symlink — that is the usual cause,
because the SkyPilot daemon does not inherit `EnvironmentFile`.

### Then prove the leak path, for about a nickel

Worth doing in the same session: the GCP orphan passes match SkyPilot's real node shape, and that
narrowing has been validated against *recorded* names but never against a live instance. One
cheap job closes it:

```bash
# laptop. The cpu profile resolves to n4-standard-4 on GCP: $0.18-0.29/hr on demand plus
# $0.0055/hr disk, so a 10-minute cap plus a few minutes of provisioning is roughly $0.05.
uv run lab submit -c "uv run experiments/<cheap-exp>.py" --backend cpu --cloud gcp --timeout 10m
uv run lab reconcile          # WHILE it runs
uv run lab wait <job_id>
uv run lab reconcile          # after teardown settles — give it a minute, teardown is async
```

What to check in the two `reconcile` reports:

| Field | While running | After teardown |
|---|---|---|
| `gcp_project` | matches the project SkyPilot launched into | same |
| `gcp_orphans` / `gcp_disk_orphans` | `[]` — the live instance is suppressed by its running cluster | `[]` |
| `gcp_unmatched` | `[]` — a non-empty list here means the node-shape predicate has drifted and is **no longer matching our own instances** | `[]` |
| exit code | 0 | 0 |

A name appearing under `gcp_unmatched` that is obviously ours is the signal that matters: it means
the leak passes would go blind and report clean. See `is_lab_cluster_node` and the
`test_the_predicate_accepts_*` tests.

## Live smoke (run once, at night, before trusting it)

1. Laptop: `lab register -c "uv run experiments/<cheap-exp>.py" --gpu RTX_4090:1 --timeout 15m \
   --max-hourly 0.30 --max-cost 0.10 --window 23:00-07:00 --tz <your-tz> --expires +1d`
2. Next morning, laptop: `lab queue list` (state `succeeded`), `lab status <job_id>` (mirrored
   manifest, cost recorded), `lab fetch <job_id>` (artifacts from R2), and `lab reconcile`
   (no orphans).
3. Kill test: while a registered job runs, `playground` reboot the droplet — within ~2 ticks
   `lab queue list` should show `supervisor respawned (adopt)` behavior and the job must still
   tear down on completion.

## Suspend when idle

`playground suspend <lab>` destroys the droplet (it bills while up). Registrations queued while
it is down launch when it returns (`Persistent=true` catches up missed ticks).
