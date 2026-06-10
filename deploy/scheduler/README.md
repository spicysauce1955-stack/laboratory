# Scheduler host deployment (spec §7)

The scheduler is **stateless**: queue entries, control, bundles, and mirrored job manifests all
live in R2, so this host can be destroyed and recreated at any time.

## Provision (playground repo)

Use the playground project's `cloud-digitalocean` backend to create the smallest droplet
(the tick is tiny and I/O-bound), then run the steps below (manually or via an Ansible role in
that repo):

1. Install uv + git; `git clone <laboratory remote> /opt/laboratory && cd /opt/laboratory && uv sync --extra skypilot`.
2. Create user `lab`; `cp deploy/scheduler/scheduler.env.example /etc/lab/scheduler.env` and fill
   in real credentials (mode 0600, owner `lab`).
3. `cp deploy/scheduler/lab-scheduler.{service,timer} /etc/systemd/system/`
4. `systemctl daemon-reload && systemctl enable --now lab-scheduler.timer`

## Verify

- `systemctl list-timers lab-scheduler.timer` — next tick scheduled.
- From the laptop: `lab queue list` — `heartbeat_age_s` under ~120.

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
