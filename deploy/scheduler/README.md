# Scheduler host deployment (spec §7)

The scheduler is **stateless**: queue entries, control, bundles, and mirrored job manifests all
live in R2, so this host can be destroyed and recreated at any time.

## Provision (playground repo)

Use the playground project's `cloud-digitalocean` backend to create the smallest droplet
(the tick is tiny and I/O-bound), then run the steps below (manually or via an Ansible role in
that repo):

1. Install uv + git; `git clone <laboratory remote> /opt/laboratory && cd /opt/laboratory && uv sync --extra skypilot --extra r2`.
   **Add `--extra gcp` if you will register `--cloud gcp` jobs** — without it the host cannot
   provision on GCP at all, and the failure only shows up at launch time, unattended, at 3am.
2. Create user `lab`; `cp deploy/scheduler/scheduler.env.example /etc/lab/scheduler.env` and fill
   in real credentials (mode 0600, owner `lab`).
3. `cp deploy/scheduler/lab-scheduler.{service,timer} /etc/systemd/system/`
4. `systemctl daemon-reload && systemctl enable --now lab-scheduler.timer`

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
- If you configured GCP: on the host, `sudo -u lab uv run sky check gcp` must show **GCP:
  enabled**. Do this *before* registering a deferred GCP job — a credential problem discovered at
  launch time wastes the whole night the job was scheduled for.

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
