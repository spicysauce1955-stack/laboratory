# Scheduler host deployment (spec §7)

The scheduler is **stateless**: queue entries, control, bundles, and mirrored job manifests all
live in R2, so this host can be destroyed and recreated at any time.

## Provision (playground repo)

Use the playground project's `cloud-digitalocean` backend to create the smallest droplet
(the tick is tiny and I/O-bound), then run the steps below (manually or via an Ansible role in
that repo).

The host runs a **pinned release of the lab** against a clone of your **experiment project** —
the same split as on your laptop. The lab is a tool; the project supplies the git history the
scheduler pins as provenance, the `runs/` it writes, and the `.env` it reads.

1. Install uv + git.
2. Install the lab at the version you want the host to run:
   ```bash
   uv tool install "laboratory[skypilot,r2] @ git+https://github.com/spicysauce1955-stack/laboratory@v0.5.0"
   ```
   **Add `gcp` to the extras if you will register `--cloud gcp` jobs** — without it the host
   cannot provision on GCP at all, and the failure only shows up at launch time, unattended, at
   3am.
3. Clone the experiment project: `git clone <project remote> /opt/tempotron-capacity`.
4. Create user `lab`; `cp deploy/scheduler/scheduler.env.example /etc/lab/scheduler.env` and fill
   in real credentials (mode 0600, owner `lab`).
5. `cp deploy/scheduler/lab-scheduler.{service,timer} /etc/systemd/system/`, then point the unit
   at the project — two lines:
   ```ini
   WorkingDirectory=/opt/tempotron-capacity
   Environment=LAB_REPO_DIR=/opt/tempotron-capacity
   ExecStart=/root/.local/bin/lab scheduler tick --backend skypilot
   ```
   (`uv tool install` puts `lab` on `~/.local/bin`; use the path for the user the unit runs as.)
6. `systemctl daemon-reload && systemctl enable --now lab-scheduler.timer`

### Upgrading the host

```bash
uv tool upgrade laboratory     # or re-run `uv tool install` pinned to the new tag
```

Nothing else moves — the project clone and `/etc/lab/scheduler.env` are untouched.

### Cutting over from a `/opt/laboratory` clone

Hosts provisioned before v0.5.0 ran `uv run lab scheduler tick` inside a clone of the *lab* repo.
To move to the layout above:

1. **Drain the queue first** — `lab queue list` must be empty, and no job in flight. The queue
   dir and `runs/` are resolved relative to the repo, so pending registrations under
   `/opt/laboratory` are stranded the moment `LAB_REPO_DIR` moves.
2. `systemctl stop lab-scheduler.timer`, then follow steps 2–6 above.
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
  lab doctor --cloud gcp               # credentials, project, billing, APIs, IAM, quota
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
